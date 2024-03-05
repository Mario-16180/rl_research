#!/bin/bash

# Set the path to your Python executable
PYTHON_EXECUTABLE=python

# Specify your Python script and its arguments
SCRIPT_PATH=experimenting/bipedal_walker_training.py
ALPHA=0.0008495181160859094
BETA=0.0007155191012605304
THETA=0.0007407432630036768
USE_CURRICULUM=False
BATCH_SIZE=256
FUNCTIONS_UPDATES=10
GAMMA=0.983864598157926
TAU=0.004828958346952071
GRAD_CLIP_VALUE=19.096100212751416
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
        --save_agent "$SAVE_AGENT" \
        --seed $SEED
done
