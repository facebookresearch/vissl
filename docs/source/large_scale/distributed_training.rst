Train on multiple-gpus
==========================

VISSL supports training any model on 1-gpu or more. Typically, a single machine can have 2, 4 or 8-gpus. If users want to train on >1 gpus within a single machine, it's very easy.
Typically for single machine training, this involves correctly setting the number of gpus to use with :code:`DISTRIBUTED.NUM_PROC_PER_NODE`.

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
      # set this to the number of gpus per machine. This ensures that each gpu of the
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
      # if True, does the gradient reduction in DDP manually. This is useful during the
      # activation checkpointing and sometimes saving the memory from the pytorch gradient
      # buckets.
      MANUAL_GRADIENT_REDUCTION: False


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
      # set this to the number of gpus per machine. This ensures that each gpu of the
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
      # if True, does the gradient reduction in DDP manually. This is useful during the
      # activation checkpointing and sometimes saving the memory from the pytorch gradient
      # buckets.
      MANUAL_GRADIENT_REDUCTION: False


Using SLURM
=============

Please follow the documentation
`here <https://github.com/facebookresearch/vissl/blob/main/docs/source/train_resource_setup.rst#train-on-slurm-cluster>`_.
