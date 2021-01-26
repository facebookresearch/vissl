Train on multiple-gpus
==========================

VISSL supports training any model on 1-gpu or more. Typically, a single machine can have 2, 4 or 8-gpus. If users want to train on >1 gpus within the single machine, it's very easy.
Typically for single machine training, this involves correctly setting the number of gpus to use via :code:`DISTRIBUTED.NUM_PROC_PER_NODE`.

The config will look like:

.. code-block:: yaml

    DISTRIBUTED:
      BACKEND: nccl           # set to "gloo" if desired
      NUM_NODES: 1            # no change needed
      NUM_PROC_PER_NODE: 2    # user sets this to number of gpus to use
      INIT_METHOD: tcp        # set to "file" if desired
      RUN_ID: auto            # Set to file_path if using file method. No change needed for tcp and a free port on machine is automatically detected.


The list of all the options exposed by VISSL:

.. code-block:: yaml

    DISTRIBUTED:
      # backend for communication across gpus. Use nccl by default. For cpu training, set
      # "gloo" as the backend.
      BACKEND: "nccl"
      # whether to output the NCCL info during training. This allows to debug how
      # nccl communication is configured.
      NCCL_DEBUG: False
      # tuning parameter to speed up all reduce by specifying number of nccl threads to use.
      # by default, we use whatever the default is set by nccl or user system.
      NCCL_SOCKET_NTHREADS: ""
      # whether model buffers are BN buffers are broadcast in every forward pass
      BROADCAST_BUFFERS: True
      # number of machines to use in training. Each machine can have many gpus. NODES count
      # number of unique hosts.
      NUM_NODES: 1
      # set this to the number of gpus per machine. This ensrures that each gpu of the
      # node has a process attached to it.
      NUM_PROC_PER_NODE: 8
      # this could be: tcp | env | file or any other pytorch supported methods
      INIT_METHOD: "tcp"
      # every training run should have a unique id. Following are the options:
      #   1. If using INIT_METHOD=env, RUN_ID="" is fine.
      #   2. If using INIT_METHOD=tcp,
      #      - if you use > 1 machine, set port yourself. RUN_ID="localhost:{port}".
      #      - If using 1 machine, set RUN_ID=auto and a free port will be automatically selected
      #   3. IF using INIT_METHOD=file, RUN_ID={file_path}
      RUN_ID: "auto"


Train on multiple machines
============================

VISSL allows scaling a training beyond 1-machine in order to speed up training. VISSL makes it extremely easy to scale up training.
Typically for single machine training, this involves correctly setting the following options:

- Number of gpus to use
- Number of nodes
- INIT_METHOD for PyTorch distributed training which determines how gpus will communicate for all reduce operations.

Putting togethe the above, if user wants to train on 2 machines where each machine has 8-gpus, the config will look like:

.. code-block:: yaml

    DISTRIBUTED:
      BACKEND: nccl
      NUM_NODES: 2               # user sets this to number of machines to use
      NUM_PROC_PER_NODE: 8       # user sets this to number of gpus to use per machine
      INIT_METHOD: tcp           # recommended if feasible otherwise
      RUN_ID: localhost:{port}   # select the free port


The list of all the options exposed by VISSL:

.. code-block:: yaml

    DISTRIBUTED:
      # backend for communication across gpus. Use nccl by default. For cpu training, set
      # "gloo" as the backend.
      BACKEND: "nccl"
      # whether to output the NCCL info during training. This allows to debug how
      # nccl communication is configured.
      NCCL_DEBUG: False
      # tuning parameter to speed up all reduce by specifying number of nccl threads to use.
      # by default, we use whatever the default is set by nccl or user system.
      NCCL_SOCKET_NTHREADS: ""
      # whether model buffers are BN buffers are broadcast in every forward pass
      BROADCAST_BUFFERS: True
      # number of machines to use in training. Each machine can have many gpus. NODES count
      # number of unique hosts.
      NUM_NODES: 1
      # set this to the number of gpus per machine. This ensrures that each gpu of the
      # node has a process attached to it.
      NUM_PROC_PER_NODE: 8
      # this could be: tcp | env | file or any other pytorch supported methods
      INIT_METHOD: "tcp"
      # every training run should have a unique id. Following are the options:
      #   1. If using INIT_METHOD=env, RUN_ID="" is fine.
      #   2. If using INIT_METHOD=tcp,
      #      - if you use > 1 machine, set port yourself. RUN_ID="localhost:{port}".
      #      - If using 1 machine, set RUN_ID=auto and a free port will be automatically selected
      #   3. IF using INIT_METHOD=file, RUN_ID={file_path}
      RUN_ID: "auto"


Using SLURM
=============

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
