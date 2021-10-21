Add new Meters
=======================

VISSL allows adding new meters easily. Follow the steps below to add a new loss:

- **Step1**: Create a new loss :code:`my_new_meter` in :code:`vissl/meters/my_new_meter.py` following the template

.. code-block:: bash

    from classy_vision.meters import ClassyMeter, register_meter

    @register_meter("my_new_meter")
    class MyNewMeter(ClassyMeter):
        """
        Add docuementation on what this meter does

        Args:
            add documentation about each meter parameter
        """

        def __init__(self, meters_config: AttrDict):
            # implement what the init method should do like
            # setting variable to update etc.
            self.reset()

        @classmethod
        def from_config(cls, meters_config: AttrDict):
            """
            Get the MyNewMeter instance from the user defined config
            """
            return cls(meters_config)

        @property
        def name(self):
            """
            Name of the meter
            """
            return "my_new_meter"

        @property
        def value(self):
            """
            Value of the meter which has been globally synced. This is the value printed and
            recorded by user.
            """
            # implement how the value should be calculated/finalized/returned to user
            ....

            return {"my_metric_name": value, ....}

        def sync_state(self):
            """
            Globally syncing the state of each meter across all the trainers.
            Should perform distributed communications like all_gather etc
            to correctly gather the global values to compute the metric
            """
            # implement what Communications should be done to globally sync the state
            ...

            # update the meter variables to store these global gathered values
            ...

        def reset(self):
            """
            Reset the meter. Should reset all the meter variables, values.
            """
            self._scores = torch.zeros(0, self.num_classes, dtype=torch.float32)
            self._targets = torch.zeros(0, self.num_classes, dtype=torch.int8)
            self._total_sample_count = torch.zeros(1)
            self._curr_sample_count = torch.zeros(1)

        def __repr__(self):
            # implement what information about meter params should be
            # printed by print(meter). This is helpful for debugging
            return repr({"name": self.name, "value": self.value})

        def set_classy_state(self, state):
            """
            Set the state of meter. This is the state loaded from a checkpoint when the model
            is resumed
            """
            # implement how to set the state of the meter
            ....

        def get_classy_state(self):
            """
            Returns the states of meter that will be checkpointed. This should include
            the variables that are global, updated and affect meter value.
            """
            return {
                "name": self.name,
                ...
            }

        def update(self, model_output, target):
            """
            Update the meter every time meter is calculated
            """
            # implement how to update the meter values
            ...

        def validate(self, model_output, target):
            """
            Validate that the input to meter is valid
            """
            # implement how to enforce the validity of the meter inputs
            ....


- **Step2**: Register the meter and meter params with VISSL Configuration. Add the params that the meter takes in
  `VISSL defaults.yaml <https://github.com/facebookresearch/vissl/blob/main/vissl/config/defaults.yaml>`_ as follows:

.. code-block:: yaml

    METERS:
      my_new_meter:
        param1: value1
        param2: value2
        ...


- **Step3**: Meter is ready to use. Simply set the configuration param :code:`METERS.name=my_new_meter` or if you want to use multiple meters :code:`METERS.names=[meter_one, my_new_meter]`.
