Add new Hooks
=======================

Hooks are the helper functions that can be executed at several parts of a training process as described below:

- :code:`on_start`: These hooks are executed before the training starts.

- :code:`on_phase_start`: executed at the beginning of every epoch (including test, train epochs)

- :code:`on_forward`: executed after every forward pass

- :code:`on_loss_and_meter`: executed after loss and meters are calculateds

- :code:`on_backward`: executed after every backward pass of the model

- :code:`on_update`: executed after model parameters are updated by the optimizer

- :code:`on_step`: executed after one single training (or test) iteration finishes

- :code:`on_phase_end`: executed after the epoch (train or test) finishes

- :code:`on_end`: executed at the very end of training.

Hooks are executed by inserting :code:`task.run_hooks(SSLClassyHookFunctions.<type>.name)` at several steps of the training.

Users can add new hooks by following the steps below:

- **Step1**: Create your new hook in :code:`vissl/hooks/my_hook.py` following the template.

.. code-block:: bash

    from classy_vision.hooks.classy_hook import ClassyHook

    class MyAwesomeNewHook(ClassyHook):
        """
        Logs the number of paramaters, forward pass FLOPs and activations of the model.
        Adapted from: https://github.com/facebookresearch/ClassyVision/blob/main/classy_vision/hooks/model_complexity_hook.py#L20    # NOQA
        """

        # define all the functions that your hook should execute. If the hook
        # executes nothing for a particular function, mark it as a noop.
        # Example: if the hook only does something for `on_start', then set:
        #    on_phase_start = ClassyHook._noop
        #    on_forward = ClassyHook._noop
        #    on_loss_and_meter = ClassyHook._noop
        #    on_backward = ClassyHook._noop
        #    on_update = ClassyHook._noop
        #    on_step = ClassyHook._noop
        #    on_phase_end = ClassyHook._noop
        #    on_end = ClassyHook._noop

        def on_start(self, task: "tasks.ClassyTask") -> None:
            # implement what your hook should execute at the beginning of training
            ...

        def on_phase_start(self, task: "tasks.ClassyTask") -> None:
            # implement what your hook should execute at the beginning of each epoch
            # (training or test epoch). Use `task.train' boolean to detect if the current
            # epoch is train or test
            ...

        def on_forward(self, task: "tasks.ClassyTask") -> None:
            # implement what your hook should execute after the model forward pass is done
            # should handle the train or test phase.
            # Use `task.train' boolean to detect if the current epoch is train or test
            ...

        def on_loss_and_meter(self, task: "tasks.ClassyTask") -> None:
            # implement what your hook should execute after the loss and meters are
            # calculated
            ...

        def on_backward(self, task: "tasks.ClassyTask") -> None:
            # implement what your hook should execute after backward pass is done. Note
            # that the model parameters are not yet updated
            ...

        def on_update(self, task: "tasks.ClassyTask") -> None:
            # implement what your hook should execute after the model parameters are updated
            # by the optimizer following LR and weight decay
            ...

        def on_step(self, task: "tasks.ClassyTask") -> None:
            # implement what your hook should execute after a training / test iteration
            # is done
            ...

        def on_phase_end(self, task: "tasks.ClassyTask") -> None:
            # implement what your hook should execute after a phase (train or test)
            # is done
            ...

        def on_end(self, task: "tasks.ClassyTask") -> None:
            # implement what your hook should execute at the end of training
            # (or testing, feature extraction)
            ...


- **Step2**: Inform VISSL on how/when to use the hook in :code:`default_hook_generator` method in :code:`vissl/hooks/__init__.py`.
  We recommend adding some configuration params like :code:`MONITOR_PERF_STATS` in :code:`vissl/config/defaults.yaml` so that
  you can set the usage of hook easily from the config file.

- **Step3**: Test your hook is working by simply running a config file and setting the config parameters you added in Step2 above.
