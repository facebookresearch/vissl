# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import io
import logging
from typing import Any, Iterable, Tuple

import torch
from classy_vision.generic.distributed_util import get_rank, get_world_size
from iopath.common.file_io import PathManager
from PIL import Image, ImageFile
from vissl.config import AttrDict
from vissl.data.data_helper import get_mean_image, QueueDataset


def create_path_manager() -> PathManager:
    # TODO: move this inline import out after AIRStore OSS public released
    from airstore.client.airstore_tabular import AIRStorePathHandler

    pathmanager = PathManager()
    pathmanager.register_handler(AIRStorePathHandler())
    pathmanager.set_strict_kwargs_checking(False)
    return pathmanager


class AirstoreDataset(QueueDataset):
    def __init__(
        self, cfg: AttrDict, data_source: str, path: str, split: str, dataset_name: str
    ):
        super(AirstoreDataset, self).__init__(
            queue_size=cfg["DATA"][split]["BATCHSIZE_PER_REPLICA"]
        )
        self.pathmanager = create_path_manager()
        self.cfg = cfg
        self.batch_size = cfg["DATA"][split]["BATCHSIZE_PER_REPLICA"]
        self.airstore_uri = path
        self.split = split
        self.epoch = 0
        self.start_iter = 0
        self.enable_queue_dataset = cfg["DATA"][self.split]["ENABLE_QUEUE_DATASET"]
        self.global_rank = get_rank()
        self.global_world_size = get_world_size()
        self._iterator = None

    def set_epoch(self, epoch: int):
        # set by trainer when train on new epoch or restore from a checkpoint
        logging.info(f"set epoch to {epoch} in airstore dataset")
        self.epoch = epoch

    def set_start_iter(self, start_iter: int):
        # set by trainer when train on restoring from a checkpoint
        logging.info(f"set start_iter to {start_iter} in airstore dataset")
        if start_iter < 0:
            raise Exception(f"{start_iter} is not a valid iteration value")
        self.start_iter = start_iter

    def _calculate_skip_samples(self, num_workers: int, worker_id: int) -> int:
        # this function is used for calcuate how many samples we should skip per
        # each worker when resuming from a checkpoint
        per_replica_skip = self.start_iter * self.batch_size
        per_worker_skip = per_replica_skip // num_workers
        # since dataloader fetching from each worker by roundrobin so we can
        # calculate exactly which worker has one extra to skip when per_replica_skip
        # can't be divided by num_workers cleanly
        if worker_id < per_replica_skip % num_workers:
            per_worker_skip += 1
        return per_worker_skip

    def _open_iterator(self) -> Iterable[Any]:
        # data iterator from airstore for current data split.
        # data are sharded by global total number of workers after shuffling

        split_cfg = self.cfg["DATA"][self.split]

        # extract numbers of dataloading workers and current worker id (range from
        # 0 to num_workers-1) from torch.utils. If we can't get worker_info we
        # assume the current process is the only dataloading worker.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            num_workers = 1
            worker_id = 0
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

        # split the dataset for each worker
        airstore_world_size = self.global_world_size * num_workers
        # each worker take it's split by it's parent process rank and worker id
        airstore_rank = self.global_rank * num_workers + worker_id

        return self.pathmanager.opent(
            self.airstore_uri,
            "r",
            skip_samples=self._calculate_skip_samples(num_workers, worker_id),
            enable_shuffle=getattr(split_cfg, "AIRSTORE_ENABLE_SHUFFLE", True),
            shuffle_window=getattr(split_cfg, "AIRSTORE_SHUFFLE_WINDOW", 128),
            seed=self.epoch,
            world_size=airstore_world_size,
            rank=airstore_rank,
            limit=getattr(split_cfg, "DATA_LIMIT", -1),
            offset=getattr(split_cfg, "DATA_OFFSET", 0),
            num_of_threads=getattr(split_cfg, "AIRSTORE_NUM_OF_THREADS", 2),
            prefetch=getattr(split_cfg, "AIRSTORE_PREFETCH", 1),
            max_holding_bundles=getattr(split_cfg, "AIRSTORE_MAX_HOLDING_BUNDLES", 5),
            bundle_download_timeout_ms=getattr(
                split_cfg, "AIRSTORE_BUNDLE_DOWNLOAD_TIMEOUT_MS", 30000
            ),
            max_retries=getattr(split_cfg, "AIRSTORE_MAX_RETRIES", 5),
            dataset_catalog_path=getattr(
                split_cfg, "AIRSTORE_DS_CATALOG_PATH", None
            ),  # temporary need during airstore development
            env=getattr(
                split_cfg, "AIRSTORE_ENV", "OSS"
            ),  # env need set to "fb" if run in FB, otherwise set to "OSS"
        )

    def num_samples(self) -> int:
        return self._open_iterator().total_size

    def __len__(self) -> int:
        return self.num_samples()

    def __getitem__(self, index) -> Tuple[Image.Image, bool]:
        if self._iterator is None:
            self._iterator = self._open_iterator()

        if not self.queue_init and self.enable_queue_dataset:
            self._init_queues()

        try:
            # TODO (wpc, prigoyal): we should check images are good when we are
            # uploading them to airstore.
            ImageFile.LOAD_TRUNCATED_IMAGES = True

            image_bytes = next(self._iterator)["image"]
            img = Image.open(io.BytesIO(image_bytes))

            if img.mode != "RGB":
                img = img.convert("RGB")

            if self.enable_queue_dataset:
                self.on_sucess(img)
            is_success = True
        except Exception as e:
            # TODO: airstore should have no failed images
            #       because they are filtered at prepare time.
            #       Then, this should be removed.
            logging.warning(e)
            is_success = False
            # if we have queue dataset class enabled, we try to use it to get
            # the seen valid images
            if self.enable_queue_dataset:
                img, is_success = self.on_failure()
                if img is None:
                    raise RuntimeError(
                        "Encountered invalid image and couldn't load from QueueDataset"
                    )
            else:
                img = get_mean_image(self.cfg["DATA"][self.split].DEFAULT_GRAY_IMG_SIZE)
        return img, is_success
