from torch.utils.data import DataLoader
from vissl.data.dataloaders.dataloader_factory_base import DataLoaderFactoryBase


class DataLoaderFactory(DataLoaderFactoryBase):
    """
    Pytorch Default data loader factory.
    """

    def get_dataloader(self):
        """
        @return: Dataloader class instance.
        """
        return DataLoader(
            dataset=self.dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            sampler=self.sampler,
            drop_last=self.drop_last,
            worker_init_fn=self.worker_init_fn,
        )
