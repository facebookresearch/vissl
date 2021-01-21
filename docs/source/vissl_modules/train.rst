Training
=================================

The training in VISSL is composed of following components: Trainer, train task and train step


Trainer
-----------

The main entry point for any training or feature extraction workflows in VISSL if the trainer. It performs following:

- The trainer constructs a :code:`train_task` which prepares all the components of the training (optimizer, loss, meters, model etc) using the settings specified by user in the yaml config file. Read below for details about train task.

- Setup the distributed training. VISSL support both GPU and CPU only training.

  - (1) Initialize the :code:`torch.distributed.init_process_group` if the distributed is not already initialized. The init_method, backend are specified by user in the yaml config file. See `VISSL defaults.yaml file <https://github.com/facebookresearch/vissl/blob/master/vissl/config/defaults.yaml>`_ for description on how to set :code:`init_method`, :code:`backend`.

  - (2) We also set the global cuda device index using torch.cuda.set_device or cpu device

- Executed the training or feature extraction workflows depending on :code`engine_name` set by users.


Training workflow
~~~~~~~~~~~~~~~~~~~~
The training workflows executes the following steps. We get the training loop to use (vissl default is :code:`standard_train_step` but the user can create their own training loop and specify the name :code:`TRAINER.TRAIN_STEP_NAME`). The training happens:

1. Execute any hooks at the start of training (mostly resets the variable like iteration num phase_num etc)

2. For each epoch (train or test), run the hooks at the start of an epoch. Mostly involves setting things like timer, setting dataloader epoch etc

3. Execute the training loop (1 training iteration) involving forward, loss, backward, optimizer update, metrics collection etc.

4. At the end of epoch, sync meters and execute hooks at the end of phase. Involves things like checkpointing model, logging timers, logging to tensorboard etc


Feature extraction workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set :code:`engine_name=extract_features` in the config file to enable feature extraction.

Extract workflow supports multi-gpu feature extraction. Since we are only extracting features, only the model is built (and initialized from some model weights file if specified by user). The model is set to the eval mode fully. The features are extracted for whatever data splits (train, val, test) etc that user wants.


Train Task
----------------------

A task prepares and holds all the components of a training like optimizer, datasets, dataloaders, losses, meters etc. Task also contains the variable like training iteration, epoch number etc. that are updated during the training.

We prepare every single component according to the parameter settings user wants and specified in the yaml config file.

Task also supports 2 additional things:

- converts the model BatchNorm layers to the synchronized batchnorm. Set the :code:`MODEL.SYNC_BN_CONFIG.CONVERT_BN_TO_SYNC_BN=true`

- sets mixed precision (apex and pytorch both supported). Set the :code:`MODEL.AMP_PARAMS.USE_AMP=true` and select the desired AMP settings.


Train Loop
-----------------

VISSL implements a default training loop (single iteration step) that is used for self-supervised training of all VISSL reference approaches, for feature extraction and for supervised workflows. Users con
implement their own training loop.

The training loop performs: data read, forward, loss computation, backward, optimizer step, parameter updates.

Various intermediate steps are also performed:

- logging the training loss, training eta, LR, etc to loggers

- logging to tensorboard,

- performing any self-supervised method specific operations (like in MoCo approach, the momentum encoder is updated), computing the scores in swav

- checkpointing model if user wants to checkpoint in the middle of an epoch

To select the training loop:

.. code-block:: yaml

    TRAINER:
      # default training loop. User can define their own loop and use that instead.
     TRAIN_STEP_NAME: "standard_train_step"
