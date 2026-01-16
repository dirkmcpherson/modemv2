#!/bin/bash

# Default values
MULTI=""
NAME="exp_name=ms_pick_cube"
SEED="seed=1"
DEMOS="demos=0"
LAUNCHER="hydra/launcher=local"
BATCH_SIZE="batch_size=256"
SEED_STEPS="seed_steps=5000"
EVAL_EPISODES="eval_episodes=10"
EVAL_FREQ="eval_freq=2500"
MIX_SCHEDULE="mix_schedule=\"linear(0.0,1.0,5000,105000)\""

# Override with command line arguments if provided
if [ $# -ge 1 ]; then
    if [ $1 = "test" ]; then
        NAME="exp_name=ms_pick_cube_test"
        BATCH_SIZE="batch_size=16"
        SEED_STEPS="seed_steps=300"
        EVAL_EPISODES="eval_episodes=3"
        EVAL_FREQ="eval_freq=300"
        MIX_SCHEDULE="mix_schedule=\"linear(0.0,1.0,300,600)\""
    fi
fi

export PYTHONPATH=$PYTHONPATH:$PWD/tasks/robohive

python train.py $MULTI \
    suite=ms \
    task=ms-PickCube-v1 \
    $NAME \
    iterations=1 \
    discount=0.95 \
    train_steps=100000 \
    $SEED \
    $DEMOS \
    $BATCH_SIZE \
    $SEED_STEPS \
    $EVAL_EPISODES \
    $EVAL_FREQ \
    $MIX_SCHEDULE \
    $LAUNCHER
