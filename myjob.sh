#!/bin/bash
#SBATCH -J pusht_z #job name
#SBATCH --time=02-00:00:00 #requested time (DD-HH:MM:SS)
#SBATCH -p gpu #running on "mpi" partition/queue
#SBATCH --gres=gpu:a100:1 #requesting 1 GPU
#SBATCH --constraint="a100-80G"
#SBATCH -N 1 #1 nodes
#SBATCH -n 32 #2 tasks total (default 1 CPU core per task) = # of cores
#SBATCH --mem=64g #RAM total
#SBATCH --output=pusht.%j.%N.out #saving standard output to file, %j=JOBID, %N=NodeName
#SBATCH --error=pusht.%j.%N.err #saving standard error to file, %j=JOBID, %N=NodeName
#SBATCH --mail-type=ALL #email optitions
#SBATCH --mail-user=james.staley625703@tufts.edu 


#[commands_you_would_like_to_exe_on_the_compute_nodes] 
# for example, running a python script
# 1st, load the modulemodule load anaconda/2021.05
# run pythonpython myscript.py #make sure myscript.py exists in the current directory or provide thefull path to script

module load anaconda/2021.11
module load cuda/12.2
sleep 10
source activate modemv2

echo "Starting training runs on $(hostname)"
nvidia-smi

#!/bin/bash

# Default values
MULTI=""
NAME="exp_name=ms_pick_cube_demos"
SEED_VAL=$((1 + RANDOM % 1000))
SEED="seed=$SEED_VAL"
DEMOS="demos=200"
LAUNCHER="hydra/launcher=local"
BATCH_SIZE="batch_size=256"
SEED_STEPS="seed_steps=5000"
EVAL_EPISODES="eval_episodes=10"
EVAL_FREQ="eval_freq=2500"
MIX_SCHEDULE="mix_schedule=\"linear(0.0,1.0,5000,105000)\""
TRAIN_STEPS="train_steps=100000"
EPISODE_LENGTH="episode_length=100"

# Override with command line arguments if provided
if [ $# -ge 1 ]; then
    if [ $1 = "test" ]; then
        NAME="exp_name=ms_pick_cube_test"
        BATCH_SIZE="batch_size=16"
        SEED_STEPS="seed_steps=300"
        EVAL_EPISODES="eval_episodes=3"
        EVAL_FREQ="eval_freq=300"
        MIX_SCHEDULE="mix_schedule=\"linear(0.0,1.0,300,600)\""
        TRAIN_STEPS="train_steps=1000"
    fi
fi

export PYTHONPATH=$PYTHONPATH:$PWD/modemv2/tasks/robohive

# Find executable
if [ -f "./.venv/bin/python" ]; then
    PYTHON="./.venv/bin/python"
else
    PYTHON="python"
fi

# Determine the correct suite override syntax (Hydra versions/configs vary)
# Try standard syntax first
echo "Starting training with suite=ms..."
ERR_LOG=$(mktemp)
# Note: we use 'tee' to show output live while also capturing it to check for specific Hydra errors
if $PYTHON modemv2/train.py $MULTI \
    suite=ms \
    task=ms-PickCube-v1 \
    $NAME \
    iterations=1 \
    discount=0.95 \
    $TRAIN_STEPS \
    $EPISODE_LENGTH \
    $SEED \
    $DEMOS \
    $BATCH_SIZE \
    $SEED_STEPS \
    $EVAL_EPISODES \
    $EVAL_FREQ \
    $MIX_SCHEDULE \
    $LAUNCHER 2> >(tee "$ERR_LOG" >&2); then
    rm "$ERR_LOG"
    exit 0
fi

if grep -q "Could not override 'suite'. Did you mean to override suite@_global_?" "$ERR_LOG"; then
    echo "Detected strict Hydra override requirements, retrying with suite@_global_=ms..."
    rm "$ERR_LOG"
    $PYTHON modemv2/train.py $MULTI \
        suite@_global_=ms \
        task=ms-PickCube-v1 \
        $NAME \
        iterations=1 \
        discount=0.95 \
        $TRAIN_STEPS \
        $EPISODE_LENGTH \
        $SEED \
        $DEMOS \
        $BATCH_SIZE \
        $SEED_STEPS \
        $EVAL_EPISODES \
        $EVAL_FREQ \
        $MIX_SCHEDULE \
        $LAUNCHER
else
    EXIT_CODE=$?
    rm "$ERR_LOG"
    exit $EXIT_CODE
fi

wait
