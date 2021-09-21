Add new Dataloader
=======================

VISSL currently supports PyTorch :code:`torch.utils.data.DataLoader`. If users would like to add a custom dataloader
of their own, we recommend the following steps.

- **Step1**: Create your custom dataloader class :code:`MyNewDataLoader` in :code:`vissl/data/my_loader.py`. The Dataloader should implement all the variables and member that PyTorch Dataloader uses.

- **Step2**: Import your new :code:`MyNewDataLoader` in :code:`vissl/data/__init__.py` and extend the function :code:`get_loader(...)` to use your :code:`MyNewDataLoader`. To control this from configuration file, we recommend users to add some config file options in :code:`vissl/defaults.yaml` file under :code:`DATA.TRAIN.dataloader_name`.

We welcome PRs following our `Contributing guidelines <https://github.com/facebookresearch/vissl/blob/main/.github/CONTRIBUTING.md>`_.
