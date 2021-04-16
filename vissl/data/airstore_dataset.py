import io
import logging

import torch
from classy_vision.generic.distributed_util import get_rank, get_world_size
from iopath.common.file_io import PathManager
from PIL import Image, ImageFile
from vissl.data.data_helper import QueueDataset, get_mean_image


def create_path_manager():
    # inline import until we have an AIRStore OSS public package
    from airstore.client.airstore_tabular import AIRStorePathHandler

    pathmgr = PathManager()
    pathmgr.register_handler(AIRStorePathHandler())
    pathmgr.set_strict_kwargs_checking(False)
    return pathmgr


class AirstoreDataset(QueueDataset):
    def __init__(self, cfg, data_source, path, split, dataset_name):
        super(AirstoreDataset, self).__init__(
            queue_size=cfg["DATA"][split]["BATCHSIZE_PER_REPLICA"]
        )
        self.pathmgr = create_path_manager()
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

    def set_epoch(self, epoch):
        # set by trainer when train on new epoch or restore from a checkpoint
        logging.info(f"set epoch to {epoch} in airstore dataset")
        self.epoch = epoch

    def set_start_iter(self, start_iter):
        # set by trainer when train on restoring from a checkpoint
        self.start_iter = start_iter

    def _open_iterator(self):
        # data iterator from airstore for current data split.
        # data are sharded by global total number of workers after shuffling

        data_cfg = self.cfg["DATA"]
        split_cfg = data_cfg[self.split]

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

        return self.pathmgr.opent(
            self.airstore_uri,
            "r",
            skip_samples=self.start_iter * self.batch_size,
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
            ), # temporary need during airstore development
            env=getattr(data_cfg, "AIRSTORE_ENV", "OSS"),
        )

    def num_samples(self):
        return self._open_iterator().total_size

    def __len__(self):
        return self.num_samples()

    def __getitem__(self, index):
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
