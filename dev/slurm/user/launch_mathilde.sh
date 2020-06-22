#!/bin/bash

######################### INPUT PARAMS ##################################
NODES=${NODES-1}
NUM_GPU=${NUM_GPU-8}
CFG=${CFG}        # config file to use in training
EXPT_NAME=${EXPT_NAME}
MEM=${MEM-450g}
CPU=${CPU-80}
BRANCH=${BRANCH-fix_BN_bug}
PARTITION=${PARTITION-priority}
COMMENT=${COMMENT-ggl4i_gcd}
GITHUB_REPO=${GITHUB_REPO-ssl_scaling}
NUM_DATA_WORKERS=${NUM_DATA_WORKERS-8}
MULTI_PROCESSING_METHOD=${MULTI_PROCESSING_METHOD-fork}


if [ "$NODES" = "1" ]; then
  SLURM_START_IDX=9
else
  SLURM_START_IDX=10
fi

EXP_ROOT_DIR="/checkpoint/$USER/3d/20200528_debug2/"

####################### SBATCH settings ####################################
URL="git@github.com:fairinternal/ssl_scaling.git"
HEADER="/private/home/$USER/temp_header"
cat > ${HEADER} <<- EOM
#!/bin/bash
#SBATCH --nodes=$NODES
#SBATCH --gpus=$NUM_GPU
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=$CPU
#SBATCH --partition=$PARTITION
#SBATCH --comment="$COMMENT"
#SBATCH --time=72:00:00
#SBATCH --signal=USR1@600
#SBATCH --open-mode=append
#SBATCH --mem=$MEM
#SBATCH --output=$EXP_ROOT_DIR/%j.out
#SBATCH --err=$EXP_ROOT_DIR/%j.err

master_node=\${SLURM_NODELIST:0:9}\${SLURM_NODELIST:$SLURM_START_IDX:4}
echo \$master_node
dist_run_id=\$master_node
EOM

####################### setup experiment dir ###################################
# create the experiments folder
RUN_SCRIPT="$EXP_ROOT_DIR/$GITHUB_REPO/tools/distributed_train.py"

# make the exp_dir and clone github code
mkdir -p "$EXP_ROOT_DIR"
cd "$EXP_ROOT_DIR" || exit
git clone "$URL" -b "$BRANCH" --single-branch
cd "$GITHUB_REPO" || exit
#git submodule update --init
SHA1=$(git rev-parse HEAD)

####################### prepare launch script ##################################
dist_port=40050

#loop
#EXPT_NAME="bs256"
#EXP="${EXP_ROOT_DIR}/$EXPT_NAME"
##rm -rf $EXP
#mkdir -p $EXP
#CHECKPOINT_DIR="$EXP/checkpoints/"
#mkdir -p "$CHECKPOINT_DIR"
#echo $SHA1>$EXP/git
#SCRIPT_PATH="${EXP}/launcher.sh"
#cp "$HEADER" "$SCRIPT_PATH"
#
#echo "export PYTHONPATH="$EXP_ROOT_DIR/$GITHUB_REPO/:$PYTHONPATH"
#dist_run_id+=":$dist_port"
#echo \$dist_run_id
#srun --label python -u $RUN_SCRIPT \
#  --node_id \$SLURM_NODEID \
#  --config_file "/private/home/mathilde/ssl_scaling/configs/mathilde_models/bs256_queuestart0.yaml" CHECKPOINT.DIR $CHECKPOINT_DIR MACHINE.NUM_DATALOADER_WORKERS $NUM_DATA_WORKERS MULTI_PROCESSING_METHOD $MULTI_PROCESSING_METHOD CHECKPOINT.APPEND_DISTR_RUN_ID False DISTRIBUTED.INIT_METHOD tcp DISTRIBUTED.RUN_ID \$dist_run_id " >> "$SCRIPT_PATH"
#
#chmod +x "$SCRIPT_PATH"
#((dist_port++))
#
## Install signal handler for automatic requeue
#trap_handler () {
#   echo "Caught signal: $1"
#   # SIGTERM must be bypassed
#   if [ "$1" = "TERM" ]; then
#       echo "bypass sigterm"
#   else
#     # Submit a new job to the queue
#     echo "Requeuing  ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID}"
#     # SLURM_JOB_ID is a unique representation of the job, equivalent
#     # to above
#     scontrol requeue "${SLURM_JOB_ID}"
#   fi
#}
#trap 'trap_handler USR1' USR1
#trap 'trap_handler TERM' TERM
#
############################ launch experiment ##################################
#sbatch --job-name="$EXPT_NAME" "$SCRIPT_PATH"
