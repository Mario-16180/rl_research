#!/bin/bash

# Set the path to your Python executable
PYTHON_EXECUTABLE=python

# Specify your Python script and its arguments
SCRIPT_PATH=experimenting/lunar_lander_training.py
EPISODES=500
BATCH_SIZE=64
CRITERION_NAME="MSE"
GRAD_CLIP_VALUE=6.095207206995203
EPSILON_DECAY=0.00001445883837274154
EPSILON_MIN=0
EPSILON_START=1.0
GAMMA=0.995
LEARNING_RATE=0.0001998897684361329
TAU=0.01
FIRST_LAYER_NEURONS=128
SECOND_LAYER_NEURONS=64
MEMORY_CAPACITY=75000
INITIAL_RANDOM_EXPERIENCES=2048
CURRICULUM=False
SAVE_AGENT=True

# Specify a list of seed values
SEEDS=(0 1 2 3 4 5 6 7 8 9 12 22 2023 420 69 1998 42 13 31 64)

# Loop through the seed values and run the Python script
for SEED in "${SEEDS[@]}"
do
    $PYTHON_EXECUTABLE $SCRIPT_PATH \
        --episodes $EPISODES \
        --batch_size $BATCH_SIZE \
        --criterion_name "$CRITERION_NAME" \
        --grad_clip_value $GRAD_CLIP_VALUE \
        --epsilon_decay $EPSILON_DECAY \
        --epsilon_min $EPSILON_MIN \
        --epsilon_start $EPSILON_START \
        --gamma $GAMMA \
        --learning_rate $LEARNING_RATE \
        --tau $TAU \
        --first_layer_neurons $FIRST_LAYER_NEURONS \
        --second_layer_neurons $SECOND_LAYER_NEURONS \
        --initial_random_experiences $INITIAL_RANDOM_EXPERIENCES \
        --curriculum "$CURRICULUM" \
	--save_agent "$SAVE_AGENT" \
        --seed $SEED
done
