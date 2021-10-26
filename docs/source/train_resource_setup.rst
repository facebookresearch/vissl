Train models on CPU
===========================

VISSL supports training any model on CPUs. Typically, this involves correctly setting the :code:`MACHINE.DEVICE=cpu` and adjusting the distributed settings accordingly. For example, the config settings will look like:

.. code-block:: yaml

    MACHINE:
      DEVICE: cpu
    DISTRIBUTED:
      BACKEND: gloo           # set to "gloo" for cpu only training
      NUM_NODES: 1            # no change needed
      NUM_PROC_PER_NODE: 2    # user sets this to number of gpus to use
      INIT_METHOD: tcp        # set to "file" if desired
      RUN_ID: auto            # Set to file_path if using file method. No change needed for tcp and a free port on machine is automatically detected.


Train anything on 1-gpu
=============================

If you have a configuration file (any vissl compatible file) for any training, that you want to run on 1-gpu only (for example: train SimCLR on 1 gpu, etc), you don't need to modify the config file. VISSL provides a helper script that takes care of all the adjustments.
This can facilitate debugging by allowing users to insert :code:`pdb` in their code.

VISSL also takes care of auto-scaling the Learning rate for various schedules (cosine, multistep, step etc.) if you have enabled the auto_scaling (see :code:`config.OPTIMIZER.param_schedulers.lr.auto_lr_scaling`). You can simply achieve this by using the :code:`low_resource_1gpu_train_wrapper.sh` script. An example usage:

.. code-block:: yaml

    cd $HOME/vissl
    ./dev/low_resource_1gpu_train_wrapper.sh config=test/integration_test/quick_swav



Train on SLURM cluster
========================

VISSL supports SLURM by default for training models. VISSL code automatically detects if the training environment is SLURM based on SLURM environment variables like :code:`SLURM_NODEID`, :code:`SLURMD_NODENAME`, :code:`SLURM_STEP_NODELIST`.

VISSL also provides a helper bash script `dev/launch_slurm.sh <https://github.com/facebookresearch/vissl/blob/main/dev/launch_slurm.sh>`_ that allows launching a given training on SLURM.
This script uses the content of the configuration to allocate the right number of nodes and GPUs on SLURM.

More precisely, the number of nodes and GPU by node to allocate is driven by the usual DISTRIBUTED training configuration:

.. code-block:: yaml

    DISTRIBUTED:
      NUM_NODES: 1            # no change needed
      NUM_PROC_PER_NODE: 2    # user sets this to number of gpus to use

While the more SLURM specific options are located in the "SLURM" configuration block:

.. code-block:: yaml

    # ----------------------------------------------------------------------------------- #
    # DISTRIBUTED TRAINING ON SLURM: Additional options for SLURM node allocation
    # (options like number of nodes and number of GPUs by node are taken from DISTRIBUTED)
    # ----------------------------------------------------------------------------------- #
    SLURM:
      # Whether or not to run the job on SLURM
      USE_SLURM: false
      # Name of the job on SLURM
      NAME: "vissl"
      # Comment of the job on SLURM
      COMMENT: "vissl job"
      # Partition of SLURM on which to run the job. This is a required field if using SLURM.
      PARTITION: ""
      # Where the logs produced by the SLURM jobs will be output
      LOG_FOLDER: "."
      # Maximum number of hours / minutes needed by the job to complete. Above this limit, the job might be pre-empted.
      TIME_HOURS: 72
      TIME_MINUTES: 0
      # Additional constraints on the hardware of the nodes to allocate (example 'volta' to select a volta GPU)
      CONSTRAINT: ""
      # GB of RAM memory to allocate for each node
      MEM_GB: 250
      # TCP port on which the workers will synchronize themselves with torch distributed
      PORT_ID: 40050
      # Number of CPUs per GPUs to request on the cluster.
      NUM_CPU_PER_PROC: 8
      # Any other parameters for slurm (e.g. account, hint, distribution, etc.,) as dictated by submitit.
      # Please see https://github.com/facebookincubator/submitit/issues/23#issuecomment-695217824.
      ADDITIONAL_PARAMETERS: {}

Users can customize these values by using the standard hydra override syntax (same as for any other item in the configuration), or can modify the script to fit their needs.

**Examples:**

To run a linear evaluation benchmark on a chosen checkpoint, on the SLURM partition named "dev", with the name "lin_eval":

.. code-block:: bash

    ./dev/launch_slurm.sh \
        config=benchmark/linear_image_classification/imagenet1k/eval_resnet_8gpu_transfer_in1k_linear \
        config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/path/to/my/checkpoint.torch \
        config.SLURM.NAME=lin_eval \
        config.SLURM.PARTITION=dev

To run a distributed training of SwAV on 8 nodes where each machine has 8 GPUs and for 100 epochs, on the default partition, with the name "swav_100ep_rn50_in1k":

.. code-block:: bash

    ./dev/launch_slurm.sh \
        config=pretrain/swav/swav_8node_resnet \
        config.OPTIMIZER.num_epochs=100 \
        config.SLURM.NAME=swav_100ep_rn50_in1k
