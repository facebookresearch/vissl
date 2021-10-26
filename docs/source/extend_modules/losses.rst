Add new Losses to VISSL
===============================

VISSL allows adding new losses easily. Follow the steps below to add a new loss:

- **Step1**: Create a new loss :code:`my_new_loss` in :code:`vissl/losses/my_new_loss.py` following the template

.. code-block:: python

    import pprint
    from classy_vision.losses import ClassyLoss, register_loss

    @register_loss("my_new_loss")
    class MyNewLoss(ClassyLoss):
        """
        Add documentation for what the loss does

        Config params:
            document what parameters should be expected for the loss in the defaults.yaml
            and how to set those params
        """

        def __init__(self, loss_config: AttrDict, device: str = "gpu"):
            super(MyNewLoss, self).__init__()

            self.loss_config = loss_config
            # implement what the init function should do
            ...

        @classmethod
        def from_config(cls, loss_config: AttrDict):
            """
            Instantiates MyNewLoss from configuration.

            Args:
                loss_config: configuration for the loss

            Returns:
                MyNewLoss instance.
            """
            return cls(loss_config)

        def __repr__(self):
            # implement what information about loss params should be
            # printed by print(loss). This is helpful for debugging
            repr_dict = {"name": self._get_name(), ....}
            return pprint.pformat(repr_dict, indent=2)

        def forward(self, output, target):
            # implement how the loss should be calculated. The output should be
            # torch.Tensor or List[torch.Tensor] and target should be torch.Tensor
            ...
            ...

            return loss


- **Step2**: Register the loss and loss params with VISSL Configuration. Add the params that the loss takes in
  `VISSL defaults.yaml <https://github.com/facebookresearch/vissl/blob/main/vissl/config/defaults.yaml>`_ as follows:

.. code-block:: yaml

    LOSS:
      my_new_loss:
        param1: value1
        param2: value2
        ...


- **Step3**: Loss is ready to use. Simply set the configuration param :code:`LOSS.name=my_new_loss`
