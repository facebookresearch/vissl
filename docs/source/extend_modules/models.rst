Add new Models
=======================

VISSL allows adding new models (head and trunks easily) and combine different trunks and heads to train a new model. Follow the steps below on how to add new heads or trunks.


Adding New Heads
------------------

To add a new model head, follow the steps:

- **Step1**: Add the new head :code:`my_new_head` under :code:`vissl/models/heads/my_new_head.py` following the template:

.. code-block:: bash

    import torch
    import torch.nn as nn
    from vissl.models.heads import register_model_head

    @register_model_head("my_new_head")
    class MyNewHead(nn.Module):
        """
        Add documentation on what this head does and also link any papers where the head is used
        """

        def __init__(self, model_config: AttrDict, param1: val, ....):
            """
            Args:
                add documentation on what are the parameters to the head
            """
            super().__init__()
            # implement what the init of head should do. Example, it can construct the layers in the head
            # like FC etc., initialize the parameters or anything else
            ....

        # the input to the model should be a torch Tensor or list of torch tensors.
        def forward(self, batch: torch.Tensor or List[torch.Tensor]):
            """
            add documentation on what the head input structure should be, shapes expected
            and what the output should be
            """
            # implement the forward pass of the head

- **Step2**: The new head is ready to use. Test it by setting the new head in the configuration file.

.. code-block:: yaml

    MODEL:
      HEAD:
        PARAMS: [
            ...
            ["my_new_head", {"param1": val, ...}]
            ...
        ]


Adding New Trunks
---------------------

To add a new trunk (a new architecture like vision transformers, etc.), follow the steps:


- **Step1**: Add your new trunk :code:`my_new_trunk` under :code:`vissl/data/trunks/my_new_trunk.py` following the template:

.. code-block:: bash

    import torch
    import torch.nn as nn
    from vissl.models.trunks import register_model_trunk

    @register_model_trunk("my_new_trunk")
    class MyNewTrunk(nn.Module):
        """
        documentation on what the trunk does and links to technical reports
        using this trunk (if applicable)
        """

        def __init__(self, model_config: AttrDict, model_name: str):
            super(MyNewTrunk, self).__init__()
            self.model_config = model_config

            # get the params trunk takes from the config
            trunk_config = self.model_config.TRUNK.MyNewTrunk

            # implement the model trunk and construct all the layers that the trunk uses
            model_layer1 = ??
            model_layer2 = ??
            ...
            ...

            # give a name to the layers of your trunk so that these features
            # can be used for other purposes: like feature extraction etc.
            # the name is fully upto user descretion. User may chose to
            # only name one layer which is the last layer of the model.
            self._feature_blocks = nn.ModuleDict(
                [
                    ("my_layer1_name", model_layer1),
                    ("my_layer1_name", model_layer2),
                    ...
                ]
            )

        def forward(
            self, x: torch.Tensor, out_feat_keys: List[str] = None
        ) -> List[torch.Tensor]:
            # implement the forward pass of the model. See the forward pass of resnext.py
            # for reference.
            # The output would be a list. The list can have one tensor (the trunk output)
            # or mutliple tensors (corresponding to several features of the trunk)
            ...
            ...

            return output

- **Step2**: Inform VISSL about the parameters of the trunk. Register the params with VISSL Configuration by adding the params in
  `VISSL defaults.yaml <https://github.com/facebookresearch/vissl/blob/master/vissl/config/defaults.yaml>`_ as follows:

.. code-block:: yaml

    MODEL:
      TRUNK:
        MyNewTrunk:
          param1: value1
          param2: value2
          ...

- **Step3**: The trunk is ready to use. Set the trunk name and params in your config file :code:`MODEL.TRUNK.NAME=my_new_trunk`
