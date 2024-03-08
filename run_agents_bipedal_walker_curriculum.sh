#!/bin/bash

# Set the path to your Python executable
PYTHON_EXECUTABLE=python

# Specify your Python script and its arguments
SCRIPT_PATH=experimenting/bipedal_walker_training.py
ALPHA=0.0007980459741313542
BETA=0.0005429533714713184
THETA=0.0004907946794604859
USE_CURRICULUM=True
BATCH_SIZE=256
FUNCTIONS_UPDATES=10
GAMMA=0.9818243370106222
TAU=0.003240528110847031
GRAD_CLIP_VALUE=6.568622222422699
MAX_TRAIN_STEPS_PER_CURRICULUM=90
ANTI_CURRICULUM=True
NUMBER_OF_CURRICULUMS=3
SAVE_AGENT=True

# Specify a list of seed values
SEEDS=(0 1 2 3 4 5 6 7 8 9 12 22 2023 420 69 1998 42 13 31 64)

# Loop through the seed values and run the Python script
for SEED in "${SEEDS[@]}"
do
    $PYTHON_EXECUTABLE $SCRIPT_PATH \
        --alpha $ALPHA \
        --beta $BETA \
        --theta $THETA \
        --use_curriculum "$USE_CURRICULUM" \
        --batch_size $BATCH_SIZE \
        --functions_updates $FUNCTIONS_UPDATES \
        --gamma $GAMMA \
        --tau $TAU \
        --grad_clip_value $GRAD_CLIP_VALUE \
        --max_train_steps_per_curriculum $MAX_TRAIN_STEPS_PER_CURRICULUM \
        --anti_curriculum "$ANTI_CURRICULUM" \
        --save_agent "$SAVE_AGENT" \
        --seed $SEED
done
