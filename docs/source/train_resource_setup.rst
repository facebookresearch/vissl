Train models on CPU
===========================

VISSL supports training any model on CPUs. Typically, this involves correctly setting the :code:`MACHINE.DEVICE=cpu` and adjusting the distributed settings accordingly.

For example, the config settings will look like:

.. code-block:: yaml

    MACHINE:
      DEVICE: cpu
    DISTRIBUTED:
      BACKEND: gloo           # set to "gloo" for cpu only trianing
      NUM_NODES: 1            # no change needed
      NUM_PROC_PER_NODE: 2    # user sets this to number of gpus to use
      INIT_METHOD: tcp        # set to "file" if desired
      RUN_ID: auto            # Set to file_path if using file method. No change needed for tcp and a free port on machine is automatically detected.


Train anything on 1-gpu
=============================

If you have a configuration file (any vissl compatible file) for any training, that you want to run on 1-gpu only (for example: train SimCLR on 1 gpu, etc), you don't need to modify the config file. VISSL provides a helper script that takes care of all the adjustments.
This can facilitate debugging by allowing users to insert :code:`pdb` in their code. VISSL also takes care of auto-scaling the Learning rate for various schedules (cosine, multistep, step etc.) if you have enabled the auto_scaling (see :code:`config.OPTIMIZER.param_schedulers.lr.auto_lr_scaling`). You can simply achieve this by using the :code:`low_resource_1gpu_train_wrapper.sh` script. An example usage:

.. code-block:: yaml

    cd $HOME/vissl
    ./dev/low_resource_1gpu_train_wrapper.sh config=test/integration_test/quick_swav



Train on SLURM cluster
========================

VISSL supports SLURM by default for training models. VISSL code automatically detects if the training environment is SLURM based on SLURM environment variables like :code:`SLURM_NODEID`, :code:`SLURMD_NODENAME`, :code:`SLURM_STEP_NODELIST`.

VISSL also provides a helper bash script `dev/launch_slurm.sh <https://github.com/facebookresearch/vissl/blob/master/dev/launch_slurm.sh>`_ that allows launching a given training on SLURM. Users can modify this script to meet their needs.

The bash script takes the following inputs:


.. code-block:: bash

    # number of machines to distribute training on
    NODES=${NODES-1}
    # number of gpus per machine to use for training
    NUM_GPU=${NUM_GPU-8}
    # gpus type: P100 | V100 | V100_32G etc. User should set this based on their machine
    GPU_TYPE=${GPU_TYPE-V100}
    # name of the training. for example: simclr_2node_resnet50_in1k. This is helpful to clearly recognize the training
    EXPT_NAME=${EXPT_NAME}
    # how much CPU memory to use
    MEM=${MEM-250g}
    # number of CPUs used for each trainer (i.e. each gpu)
    CPU=${CPU-8}
    # directory where all the training artifacts like checkpoints etc will be written
    OUTPUT_DIR=${OUTPUT_DIR}
    # partition of the cluster on which training should run. User should determine this parameter for their cluster
    PARTITION=${PARTITION-learnfair}
    # any helpful comment that slurm dashboard can display
    COMMENT=${COMMENT-vissl_training}
    GITHUB_REPO=${GITHUB_REPO-vissl}
    # what branch of VISSL should be used. specify your custom branch
    BRANCH=${BRANCH-master}
    # automatically determined and used for distributed training.
    # each training run must have a unique id and vissl defaults to date
    RUN_ID=$(date +'%Y%m%d')
    # number of dataloader workers to use per gpu
    NUM_DATA_WORKERS=${NUM_DATA_WORKERS-8}
    # multi-processing method to use in PyTorch. Options: forkserver | fork | spawn
    MULTI_PROCESSING_METHOD=${MULTI_PROCESSING_METHOD-forkserver}
    # specify the training configuration to run. For example: to train swav for 100epochs
    # config=pretrain/swav/swav_8node_resnet config.OPTIMIZER.num_epochs=100
    CFG=( "$@" )


To run the script for training SwAV on 8 machines where each machine has 8-gpus and for 100epochs, the script can be run as:


.. code-block:: bash

    cd $HOME/vissl && NODES=8 \
      NUM_GPU=8 \
      GPU_TYPE=V100 \
      MEM=200g \
      CPU=8 \
      EXPT_NAME=swav_100ep_rn50_in1k \
      OUTPUT_DIR=/tmp/swav/ \
      PARTITION=learnfair \
      BRANCH=master \
      NUM_DATA_WORKERS=4 \
      MULTI_PROCESSING_METHOD=forkserver \
      ./dev/launch_slurm.sh \
      config=pretrain/swav/swav_8node_resnet config.OPTIMIZER.num_epochs=100
