Using Tensorboard in VISSL
==================================

VISSL provides integration of `Tensorboard <https://www.tensorflow.org/tensorboard>`_ to facilitate self-supervised training and experimentation. VISSL logs many useful metrics to Tensorboard that provide useful insights into an ongoing training:

- **Scalars**:
    - Training Loss
    - Learning Rate
    - Average Training iteration time
    - Batch size per gpu
    - Number of images per sec per gpu
    - Training ETA
    - GPU memory used
    - Peak GPU memory allocated

- **Non-scalars**:
    - Model parameters (at the start of every epoch and/or after N iterations)
    - Model parameter gradients (at the start of every epoch and/or after N iterations)


How to use Tensorboard in VISSL
--------------------------------

Using Tensorboard is very easy in VISSL and can be achieved by setting some configuration options. User needs to set :code:`HOOKS.TENSORBOARD_SETUP.USE_TENSORBOARD=true` and adjust the values of other config parameters as desired. Full set of
parameters exposed by VISSL for Tensorboard:

.. code-block:: yaml

    HOOKS:
        TENSORBOARD_SETUP:
        # whether to use tensorboard for the visualization
        USE_TENSORBOARD: False
        # log directory for tensorboard events
        LOG_DIR: "."
        EXPERIMENT_LOG_DIR: "tensorboard"
        # flush logs every n minutes
        FLUSH_EVERY_N_MIN: 5
        # whether to log the model parameters to tensorboard
        LOG_PARAMS: True
        # whether ttp log the model parameters gradients to tensorboard
        LOG_PARAMS_GRADIENTS: True
        # if we want to log the model parameters every few iterations, set the iteration
        # frequency. -1 means the params will be logged only at the end of epochs.
        LOG_PARAMS_EVERY_N_ITERS: 310

.. note::

    Please install tensorboard manually: if pip environment: :code:`pip install tensorboard` or if using conda and you prefer the conda install of tensorboard:  :code:`conda install -c conda-forge tensorboard`.

Example usage
---------------

For example, to use Tensorboard during SwAV training, the command would look like:

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/swav/swav_8node_resnet \
        config.HOOKS.TENSORBOARD_SETUP.USE_TENSORBOARD=true \
        config.HOOKS.TENSORBOARD_SETUP.LOG_PARAMS=true \
        config.HOOKS.TENSORBOARD_SETUP.LOG_PARAMS_GRADIENTS=true \
        config.HOOKS.TENSORBOARD_SETUP.LOG_DIR=/tmp/swav_tensorboard_events/
