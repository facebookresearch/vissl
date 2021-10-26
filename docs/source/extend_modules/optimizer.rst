Add new Optimizers
=======================

VISSL makes it easy to add new optimizers. VISSL depends on `ClassyVision <https://github.com/facebookresearch/ClassyVision>`_ and its optimizers.

Follow the steps below to add a new optimizer to VISSL.

- **Step1**: Create your new optimizer :code:`my_new_optimizer` under :code:`vissl/optimizers/my_new_optimizer.py` following the template:

.. code-block:: python

    from classy_vision.optim import ClassyOptimizer, register_optimizer

    @register_optimizer("my_new_optimizer")
    class MyNewOptimizer(ClassyOptimizer):
        """
        Add documentation on how the optimizer optimizes and also
        link to any papers or techincal reports that propose/use the
        the optimizer (if applicable)
        """
        def __init__(self, param1, param2, ...):
            super().__init__()
            # implement what the optimizer init should do and what variable it should store
            ...


        def prepare(self, param_groups):
            """
            Prepares the optimizer for training.

            Deriving classes should initialize the underlying PyTorch
            :class:`torch.optim.Optimizer` in this call. The param_groups argument
            follows the same format supported by PyTorch (list of parameters, or
            list of param group dictionaries).

            Warning:
                This should called only after the model has been moved to the correct
                device (gpu or cpu).
            """

        @classmethod
        def from_config(cls, config: Dict[str, Any]) -> "SGD":
            """
            Instantiates a MyNewOptimizer from a configuration.

            Args:
                config: A configuration for a MyNewOptimizer.
                    See :func:`__init__` for parameters expected in the config.

            Returns:
                A MyNewOptimizer instance.
            """

- **Step2**: Enable the automatic import of all the modules. Add the following lines of code to :code:`vissl/optimizers/__init__.py`. Skip this step if already exists.

.. code-block:: python

    from pathlib import Path
    from classy_vision.generic.registry_utils import import_all_modules

    FILE_ROOT = Path(__file__).parent

    # automatically import any Python files in the optimizers/ directory
    import_all_modules(FILE_ROOT, "vissl.optimizers")

- **Step3**: Enable the registry of the new optimizers in VISSL. Add the following line to :code:`vissl/trainer/__init__.py`. Skip this step if already exists.

.. code-block:: python

    import vissl.optimizers # NOQA


- **Step4**: The optimizer is now ready to use. Set the configuration param :code:`OPTIMIZER.name=my_new_optimizer` and set the values of the params this optimizer takes.


Add new LR schedulers
=========================

VISSL allows adding new Learning rate schedulers easily. Follow the steps below:

- **Step1**: Create a class for your :code:`my_new_lr_scheduler` under :code:`vissl/optimizers/param_scheduler/my_new_lr_scheduler.py` following the template:

.. code-block:: python

    from classy_vision.optim.param_scheduler import (
        ClassyParamScheduler,
        UpdateInterval,
        register_param_scheduler,
    )

    @register_param_scheduler("my_new_lr_scheduler")
    class MyNewLRScheduler(ClassyParamScheduler):
        """
        Add documentation on how the LR schedulers works and also
        link to any papers or techincal reports that propose/use the
        the scheduler (if applicable)

        Args:
            document all the inputs that the scheduler takes

        Example:
            show one example of how to use the lr scheduler
        """

        def __init__(
            self, param1, param2, ... , update_interval: UpdateInterval = UpdateInterval.STEP
        ):

            super().__init__(update_interval=update_interval)

            # implement what the init of LR scheduler should do, any variables
            # to initialize etc.
            ...
            ...

        @classmethod
        def from_config(cls, config: Dict[str, Any]) -> "MyNewLRScheduler":
            """
            Instantiates a MyNewLRScheduler from a configuration.

            Args:
                config: A configuration for a MyNewLRScheduler.
                    See :func:`__init__` for parameters expected in the config.

            Returns:
                A MyNewLRScheduler instance.
            """
            return cls(param1=config.param1, param2=config.param2, ...)

        def __call__(self, where: float):
            # implement what the LR value should be give the `where' which indicates
            # how far the training is. `where' values are [0, 1)
            ...
            ...

            return lr_value

- **Step2**: The new LR scheduler is ready to use. Give it a try by setting configuration param :code:`OPTIMIZER.param_schedulers.lr.name=my_new_lr_scheduler`.
