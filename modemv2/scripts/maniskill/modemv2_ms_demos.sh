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

./.venv/bin/python modemv2/train.py $MULTI \
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
