#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

######################### INPUT PARAMS ##################################
NODES=${NODES-2}    # number of machines to distribute training on
GPUS_PER_NODE=${GPUS_PER_NODE-8}
#GPU_TYPE=${GPU_TYPE-V100}
EXPT_NAME=${EXPT_NAME}
MEM=${MEM-512g}
CPU=${CPU-80}
OUTPUT_DIR=${OUTPUT_DIR}
PARTITION=${PARTITION-learnfair}
COMMENT=${COMMENT-ggl4i_gcd}
GITHUB_REPO=${GITHUB_REPO-vissl}
BRANCH=${BRANCH-vision_transformer}
RUN_ID=$(date +'%Y%m%d')
NUM_DATA_WORKERS=${NUM_DATA_WORKERS-8}
MULTI_PROCESSING_METHOD=${MULTI_PROCESSING_METHOD-forkserver}

CFG=( "$@" )

echo "$CFG"
if [ "$NODES" = "1" ]; then
  SLURM_START_IDX=9
else
  SLURM_START_IDX=10
fi

EXP_ROOT_DIR="/checkpoint/$USER/${GITHUB_REPO}/${RUN_ID}_${BRANCH}/$EXPT_NAME/"

echo $SLURM_START_IDX
####################### SBATCH settings ####################################
URL="git@github.com:facebookresearch/vissl.git"
HEADER="/private/home/$USER/temp_header"
cat > ${HEADER} <<- EOM
#!/bin/bash
#SBATCH --nodes=$NODES
#SBATCH --gpus-per-node=$GPUS_PER_NODE
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=$CPU
#SBATCH --partition=$PARTITION
#SBATCH --comment="$COMMENT"
#SBATCH --time=48:00:00
#SBATCH --signal=USR1@600
#SBATCH --open-mode=append
#SBATCH --mem=$MEM
#SBATCH --output=$EXP_ROOT_DIR/%j.out
#SBATCH --err=$EXP_ROOT_DIR/%j.err

master_node=\${SLURM_NODELIST:0:9}\${SLURM_NODELIST:$SLURM_START_IDX:4}
echo \$master_node
dist_run_id=\$master_node
EOM

echo "Master node: ${master_node}"
echo "HEADER: $HEADER"

####################### setup experiment dir ###################################
# create the experiments folder
RUN_SCRIPT="$EXP_ROOT_DIR/$GITHUB_REPO/tools/run_distributed_engines.py"
CHECKPOINT_DIR="$EXP_ROOT_DIR/checkpoints/"

echo "EXP_ROOT_DIR: $EXP_ROOT_DIR"
echo "RUN_SCRIPT: $RUN_SCRIPT"
echo "CHECKPOINT_DIR: $CHECKPOINT_DIR"

# make the exp_dir and clone github code and relevant branch
rm -rf $EXP_ROOT_DIR
mkdir -p "$EXP_ROOT_DIR"
mkdir -p "$CHECKPOINT_DIR"
cd "$EXP_ROOT_DIR" || exit
git clone "$URL" -b "$BRANCH" --single-branch
cd "$GITHUB_REPO" || exit
git submodule update --init
SHA1=$(git rev-parse HEAD)
echo "$SHA1">"$EXP_ROOT_DIR"/git

####################### prepare launch script ##################################
dist_port=40050
((dist_port++))

SCRIPT_PATH="$EXP_ROOT_DIR/launcher.sh"
cp "$HEADER" "$SCRIPT_PATH"

echo "export PYTHONPATH="$EXP_ROOT_DIR/$GITHUB_REPO/:$PYTHONPATH"
dist_run_id+=":$dist_port"
echo \$dist_run_id
srun --label python -u $RUN_SCRIPT \
  hydra.run.dir=$CHECKPOINT_DIR \
  config=${CFG[*]} \
  config.CHECKPOINT.DIR=$CHECKPOINT_DIR \
  config.DATA.NUM_DATALOADER_WORKERS=$NUM_DATA_WORKERS \
  config.MULTI_PROCESSING_METHOD=$MULTI_PROCESSING_METHOD \
  config.DISTRIBUTED.INIT_METHOD=tcp \
  hydra.verbose=True \
  config.DISTRIBUTED.RUN_ID=\$dist_run_id " >> "$SCRIPT_PATH"

chmod +x "$SCRIPT_PATH"
((dist_port++))

########################### setup trap handler ##################################

# Install signal handler for automatic requeue
trap_handler () {
   echo "Caught signal: $1"
   # SIGTERM must be bypassed
   if [ "$1" = "TERM" ]; then
       echo "bypass sigterm"
   else
     # Submit a new job to the queue
     echo "Requeuing  ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID}"
     # SLURM_JOB_ID is a unique representation of the job, equivalent
     # to above
     scontrol requeue "${SLURM_JOB_ID}"
   fi
}
trap 'trap_handler USR1' USR1
trap 'trap_handler TERM' TERM

########################### launch experiment ##################################
sbatch --job-name="$EXPT_NAME" "$SCRIPT_PATH"
