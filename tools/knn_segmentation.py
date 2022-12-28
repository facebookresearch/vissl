# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
KNN at the patch level for architectures such as ViT or XCiT
to do semantic segmentation of images
"""

import sys
from typing import Any, List

import faiss
import numpy as np
import sklearn.metrics
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
from vissl.data import dataset_catalog
from vissl.models import build_model
from vissl.utils.checkpoint import CheckpointLoader
from vissl.utils.hydra_config import compose_hydra_configuration, convert_to_attrdict
from vissl.utils.knn_utils import get_sorted_predictions


def load_dataset(config, split: str):
    """
    Load the segmentation dataset
    """
    image_transforms = transforms.Compose(
        [
            transforms.CenterCrop(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True
            ),
        ]
    )
    target_transforms = transforms.Compose(
        [
            transforms.CenterCrop(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x * 255).to(dtype=torch.int64)),
            transforms.Lambda(
                # TODO - fix this - use an ignore index instead
                lambda x: torch.where(x != 255, x, torch.tensor(0, dtype=torch.int64))
            ),
        ]
    )
    data_files, label_files = dataset_catalog.get_data_files(
        split="TRAIN" if split == "train" else "TEST",  # VISSL split name
        dataset_config=config["DATA"],
    )
    return datasets.VOCSegmentation(
        root=data_files[0],
        image_set=split,
        transform=image_transforms,
        target_transform=target_transforms,
    )


def extract_embeddings(model, data_loader, normalize_features: bool = False):
    embeddings = []
    targets = []
    full_targets = []

    for x, y in tqdm(data_loader):
        x = x.cuda()
        B, C, H, W = x.shape

        # Forward input to model to capture last feature map
        y_hat = model.trunk.forward(x, ["lastMAP"])[0]
        if normalize_features:
            y_hat = F.normalize(y_hat, dim=-1)

        # Reshape the annotation to have same shape as feature map (B, H, W, C)
        b, h, w, c = y_hat.shape
        y_target = F.interpolate(y.float(), size=(h, w), mode="nearest").int()
        y_target = y_target.permute((0, 2, 3, 1))
        assert y_hat.shape[:-1] == y_target.shape[:-1]
        y = y.permute((0, 2, 3, 1))
        assert y.shape[:-1] == torch.Size([B, H, W])

        # Now, only keep the last dimension to build an table of
        # pairs (embedding, corresponding class)
        y_hat = y_hat.reshape(
            (b * h * w, -1)
        )  # TODO - do that after, only to create the index?
        y_target = y_target.reshape((b * h * w, -1))
        embeddings.append(y_hat.cpu())
        targets.append(y_target.cpu())
        full_targets.append(y.cpu())

    # Combine all the embedding and targets together
    embeddings = torch.cat(embeddings, dim=0)
    targets = torch.cat(targets, dim=0).squeeze(-1)
    full_targets = torch.cat(full_targets, dim=0).squeeze(-1)
    return embeddings, targets, full_targets


@torch.no_grad()
def main(args, config):

    # TODO - add the low shot part (one example of each?)
    # TODO - use configuration instead
    num_neighbors = 10
    metric_type = "cosine"
    normalize_features = True
    cosine_temperature = 0.1
    output_path = "/checkpoint/qduval/temp/"

    # Load the dataset
    train_batch_size = config.DATA.TRAIN.BATCHSIZE_PER_REPLICA
    train_dataset = load_dataset(config, split="train")
    train_data_loader = data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=False
    )
    val_batch_size = config.DATA.TEST.BATCHSIZE_PER_REPLICA
    val_dataset = load_dataset(config, split="val")
    val_data_loader = data.DataLoader(
        val_dataset, batch_size=val_batch_size, shuffle=False
    )

    # Build the model and load the weights from the checkpoint
    model = build_model(config["MODEL"], config["OPTIMIZER"]).cuda()
    checkpoint = CheckpointLoader.load_and_broadcast_init_weights(
        checkpoint_path=config.MODEL.WEIGHTS_INIT.PARAMS_FILE,
        device=torch.device("cpu"),
    )
    model.init_model_from_weights_params_file(config, checkpoint)

    # Iterate over the dataset and build the a table of pairs
    # (representation of patch, corresponding dominant class)

    print("Extracting train embeddings...")
    train_embeddings, train_targets, _ = extract_embeddings(
        model, train_data_loader, normalize_features=normalize_features
    )

    print("Extracting val embeddings...")
    val_embeddings, val_targets, val_full_targets = extract_embeddings(
        model, val_data_loader, normalize_features=normalize_features
    )

    print("Building the KNN index...")
    train_keys = train_embeddings.numpy()
    if metric_type == "cosine":
        index = faiss.IndexFlatIP(train_embeddings.shape[-1])
    else:
        index = faiss.IndexFlatL2(train_embeddings.shape[-1])
    index.add(train_keys)

    print("Computing predictions for validation set...")
    val_keys = val_embeddings.numpy()

    if metric_type == "l2":
        assert num_neighbors == 1, "L2 only support 1 neighbor"
        distances, indices = index.search(val_keys, 1)
        predictions = train_targets[indices].squeeze(-1)
    else:
        assert num_neighbors > 0
        distances, indices = index.search(val_keys, num_neighbors)
        retrieved_neighbors = train_targets[indices].view(-1, 1).long()
        num_examples = val_embeddings.shape[0]
        num_classes = val_targets.max().item() + 1
        retrieval_one_hot = torch.zeros(num_examples * num_neighbors, num_classes)
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        predictions = get_sorted_predictions(
            batch_size=num_examples,
            num_classes=num_classes,
            distances=torch.from_numpy(distances),
            retrieval_one_hot=retrieval_one_hot,
            temperature=cosine_temperature,
        )
        predictions = predictions.narrow(dim=1, start=0, length=1)  # Top-1

    print(predictions.shape, val_targets.shape, val_full_targets.shape)

    print("Saving the predictions...")
    # TODO - do not hard-code the size of the patches
    np.save(f"{output_path}/val_predictions.npy", predictions.reshape(-1, 14, 14))
    np.save(f"{output_path}/val_targets.npy", val_targets.reshape(-1, 14, 14))

    print("Computing metrics...")
    predictions = predictions.numpy()
    val_targets = val_targets.numpy()
    accuracy = sklearn.metrics.accuracy_score(predictions, val_targets)
    jaccard_score = sklearn.metrics.jaccard_score(
        predictions, val_targets, average=None
    )
    jaccard_score_weighted = sklearn.metrics.jaccard_score(
        predictions, val_targets, average="weighted"
    )
    average_jaccard_score = np.mean(jaccard_score)
    print(f"Accuracy: {accuracy}")
    print(f"Jaccard Score: {jaccard_score}")
    print(f"Jaccard Score (Mean): {average_jaccard_score}")
    print(f"Jaccard Score (Mean Weighted): {jaccard_score_weighted}")

    print("Computing full resolution metrics...")
    # TODO - do not hard-code the size of the patches
    high_res_predictions = F.interpolate(
        torch.from_numpy(predictions).reshape((-1, 1, 14, 14)).float(),
        size=(val_full_targets.shape[-2], val_full_targets.shape[-1]),
        mode="bilinear",
    ).long()
    jaccard_score_weighted = sklearn.metrics.jaccard_score(
        high_res_predictions.reshape((-1,)).numpy(),
        val_full_targets.reshape((-1,)).numpy(),
        average="weighted",
    )
    print(f"Jaccard Score (Mean Weighted): {jaccard_score_weighted}")

    print("Saving the full resolution predictions...")
    np.save(
        f"{output_path}/val_full_predictions.npy",
        high_res_predictions.squeeze(1).numpy(),
    )
    np.save(f"{output_path}/val_full_targets.npy", val_full_targets)

    print("Computing full resolution binary metrics...")
    jaccard_score_weighted = sklearn.metrics.jaccard_score(
        high_res_predictions.reshape((-1,)).numpy() != 0,
        val_full_targets.reshape((-1,)).numpy() != 0,
        average="weighted",
    )
    print(f"Jaccard Score (Mean Weighted): {jaccard_score_weighted}")


def hydra_main(overrides: List[Any]):
    cfg = compose_hydra_configuration(overrides)
    args, config = convert_to_attrdict(cfg)
    main(args, config)


if __name__ == "__main__":
    """
    python tools/knn_segmentation.py config=debugging/benchmark/knn_segmentation/dino_deits16 config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/checkpoint/qduval/vissl/dino/deit_s16_job53356364_model_final_checkpoint_phase299.torch
    python tools/knn_segmentation.py config=debugging/benchmark/knn_segmentation/dino_xcit_m24_p16 config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/checkpoint/qduval/vissl/dino/xcit_24_model_final_checkpoint_phase299.torch
    """
    overrides = sys.argv[1:]
    hydra_main(overrides=overrides)
