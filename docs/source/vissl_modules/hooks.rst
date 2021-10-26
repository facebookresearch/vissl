Hooks
===============================

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


VISSL Hooks
-------------------------------------

Hooks are executed by inserting :code:`task.run_hooks(SSLClassyHookFunctions.<type>.name)` at several steps of the training. VISSL currently supports the following hooks. To see comprehensive documentation on these hooks, pelase see the :code:`defaults.yaml`.


This hook will log configured metric to Tensorboard. To enable this hook, set :code:`HOOKS.TENSORBOARD_SETUP.USE_TENSORBOARD=true` and configure the tensorboard settings.


Performance Stats hook
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This hook will log performance stats to the :code:`log.txt` output file. To enable this hook, set :code:`HOOKS.PERF_STATS.MONITOR_PERF_STATS=true` and configure the performance stats frequency and other settings.


Memory Summary hook
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This hook will log cpu and gpu memory metrics to the :code:`log.txt` output file. To enable this hook, set :code:`HOOKS.MEMORY_SUMMARY.PRINT_MEMORY_SUMMARY=true` and configure the performance stats frequency and other settings.


Model Complexity Hook
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This hook performs one single forward pass of the model on the synthetic input and computes the #FLOPs, #params and #activations in the model. To enable this hook, set :code:`HOOKS.MODEL_COMPLEXITY.COMPUTE_COMPLEXITY=true` and configure it.


Monitor Activation Statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This hook reports several activation statistics, like mean and spread, to Tensorboard. To enable this hook, set :code:`HOOKS.MONITORING.MONITOR_ACTIVATION_STATISTICS=NUM_ITERS` and configure the INPUT_SHAPE.


Profiling Hook
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This hook reports comprehensive memory and runtime profiling metrics and visualizations. To enable this hook, set :code:`HOOKS.PROFILING.MEMORY_PROFILING.TRACK_BY_LAYER_MEMORY=true` and/or :code:`HOOKS.PROFILING.RUNTIME_PROFILING.USE_PROFILER=true` and configure the additional settings as desired.


Logging, checkpoint, training variable update hooks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These hooks are used by default in VISSL and perform operations like logging the training progress (loss, LR, eta etc) on stdout, save checkpoints etc.


Self-supervised Loss hooks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

VISSL has hooks specific to self-supervised approaches like MoCo, SwAV etc. These hooks are handy in performing some intermediate operations required in self-supervision. For example: :code:`MoCoHook` is called after every forward pass of the model and updates the momentum encoder network. Users don't need to do anything special to use these hooks. If the user configuration file has the loss function for an approach, VISSL will automatically enable the hooks for the approach.
