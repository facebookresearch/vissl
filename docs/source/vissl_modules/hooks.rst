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

Hooks are executed by inserting :code:`task.run_hooks(SSLClassyHookFunctions.<type>.name)` at several steps of the training.

How to enable certain hooks in VISSL
-------------------------------------

VISSL supports many hooks. Users can configure which hooks to use from simple configuration files. The hooks in VISSL can be categorized into following buckets:

- :code:`Tensorboard hook`: to enable this hook, set :code:`TENSORBOARD_SETUP.USE_TENSORBOARD=true` and configure the tensorboard settings

- :code:`Model Complexity hook`: this hook performs one single forward pass of the model on the synthetic input and computes the #FLOPs, #params and #activations in the model. To enable this hook, set :code:`HOOKS.MODEL_COMPLEXITY.COMPUTE_COMPLEXITY=true` and configure it.

- :code:`Self-supervised Loss hooks`: VISSL has hooks specific to self-supervised approaches like MoCo, SwAV etc. These hooks are handy in performing some intermediate operations required in self-supervision. For example: :code:`MoCoHook` is called after every forward pass of the model and updates the momentum encoder network. Users don't need to do anything special for using these hooks. If the user configuration file has the loss function for an approach, VISSL will automatically enable the hooks for the approach.

- :code:`Logging, checkpoint, training variable update hooks`: These hooks are used by default in VISSL and perform operations like logging the training progress (loss, LR, eta etc) on stdout, save checkpoints etc.
