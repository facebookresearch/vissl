Add custom Train loop
=======================

VISSL implements a default training loop (single iteration step) that is used for self-supervised training of all VISSL reference approaches, for feature extraction and for supervised workflows. Users can
implement their own training loop.

The training loop performs: data read, forward, loss computation, backward, optimizer step, parameter updates.

Various intermediate steps are also performed:

- logging the training loss, training eta, LR, etc to loggers.

- logging metrics to tensorboard.

- performing any self-supervised method specific operations (like in MoCo approach, the momentum encoder is updated), computing the scores in swav.

- checkpointing model if user wants to checkpoint in the middle of an epoch.

Users can implement their custom training loop by following the steps:

- **Step1**: Create your :code:`my_new_training_loop` module under :code:`vissl/trainer/train_steps/my_new_training_loop.py` following the template:

.. code-block:: python

    from vissl.trainer.train_steps import register_train_step

    @register_train_step("my_new_training_loop")
    def my_new_training_loop(task):
        """
        add documentation on what this training loop does and how it varies from
        standard training loop in vissl.
        """
        # implement the training loop. It should take care of running the dataloader
        # iterator to get the input sample
        ...
        ...

        return task


- **Step2**: New train loop is ready to use. Set the :code:`TRAINER.TRAIN_STEP_NAME=my_new_training_loop`
