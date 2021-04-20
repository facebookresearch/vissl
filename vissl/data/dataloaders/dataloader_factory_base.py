from abc import ABC


class DataLoaderFactoryBase(ABC):
    """
    Abstract baseclass for dataloader factories. This abstraction is used to clean up the
    fetching of the dataloaders, as different dataloaders require different arguments.

    To implement a factory inherit from this class(DataLoaderFactoryBase) and overwrite the
    #get_dataloader method. Any additional arguments needed in #get_dataloader should be added in
    DataLoaderFactoryBase#__init__.
    """

    def __init__(
        self,
        dataloader_config,
        dataset_config,
        dataset,
        num_workers,
        pin_memory,
        shuffle,
        batch_size,
        collate_fn,
        sampler,
        drop_last,
        worker_init_fn,
    ):
        self.dataloader_config = dataloader_config
        self.dataset_config = dataset_config
        self.dataset = dataset
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.sampler = sampler
        self.drop_last = drop_last
        self.worker_init_fn = worker_init_fn

        self._validate_datasets()

    def get_dataloader(self):
        """
        Required method for dataloader factory class.

        @return: Dataloader class instance.
        """
        raise NotImplementedError

    def _validate_datasets(self):
        """
        Validates that the data loader supports the dataset listed.
        """
        supported_datasets = self.dataloader_config["datasets"]
        dataloader = self.dataloader_config["factory_class"]
        for dataset in self.dataset_config["DATA_SOURCES"]:
            assert (
                dataset in supported_datasets
            ), f"Dataset: { dataset } is not supported in data loader factory: { dataloader }"
