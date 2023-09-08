import torch
from collections import OrderedDict
import sys


def download_and_save_deit3():
    checkpoint = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/deit/deit_3_base_224_1k.pth",
                                                    map_location='cpu', check_hash=True)
    new_checkpoint = []
    for k in checkpoint["model"]:
        if not k.startswith("head."):
            new_k = "trunk." + k
            new_checkpoint.append([new_k, checkpoint["model"][k]])
        else:
            new_k = "heads.0.clf.0." + k[len("head."):]
            new_checkpoint.append([new_k, checkpoint["model"][k]])

    new_checkpoint = OrderedDict(new_checkpoint)

    print(new_checkpoint.keys())
    save = {"model": new_checkpoint}
    torch.save(save, "checkpoints/deitb.pth")


def download_and_save_ibot():
    checkpoint = torch.hub.load_state_dict_from_url(
        "https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitb_16/checkpoint_teacher.pth",
        map_location='cpu', check_hash=True)
    new_checkpoint = []
    for k in checkpoint["state_dict"]:
        new_k = "trunk." + k
        new_checkpoint.append([new_k, checkpoint["state_dict"][k]])

    new_checkpoint = OrderedDict(new_checkpoint)

    new_checkpoint.keys()

    save = {"model": new_checkpoint}
    torch.save(save, "checkpoints/trunk_only/ibotb.pth")


def download_and_save_dino():
    checkpoint = torch.hub.load_state_dict_from_url(
        'https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth', map_location='cpu',
        check_hash=True)

    new_checkpoint = []

    for k in checkpoint:
        if not k.startswith("head."):
            new_k = "trunk." + k
            new_checkpoint.append([new_k, checkpoint[k]])

    new_checkpoint = OrderedDict(new_checkpoint)

    print(new_checkpoint.keys())

    save = {"model": new_checkpoint}
    torch.save(save, "checkpoints/trunk_only/dino.pth")


def download_and_save_moco3():
    checkpoint = torch.hub.load_state_dict_from_url(
        'https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar', map_location='cpu',
        check_hash=True)

    new_checkpoint = []

    for k in checkpoint["state_dict"]:
        if k.startswith("module.momentum_encoder") and not k.startswith("module.momentum_encoder.head"):
            new_k = "trunk." + k[len('module.momentum_encoder.'):]
            new_checkpoint.append([new_k, checkpoint["state_dict"][k]])

    new_checkpoint = OrderedDict(new_checkpoint)

    print(new_checkpoint.keys())

    save = {"model": new_checkpoint}
    torch.save(save, "checkpoints/trunk_only/moco3.pth")


models_funs = {
    "deit3": download_and_save_deit3,
    "ibot": download_and_save_ibot,
    "dino": download_and_save_dino,
    "moco3": download_and_save_moco3
}


def main(model):
    # Your code here
    print("Downloading and saving:", model)

    models_funs[model]()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <model>")
        sys.exit(1)

    input_string = sys.argv[1]
    main(input_string)