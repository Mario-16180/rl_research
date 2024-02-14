#!/bin/bash

# Set the path to your Python executable
PYTHON_EXECUTABLE=python

# Specify your Python script and its arguments
SCRIPT_PATH=experimenting/lunar_lander_training.py
BATCH_SIZE=64
CRITERION_NAME="MSE"
GRAD_CLIP_VALUE=8.659975570109234
EPSILON_DECAY=0.00001465369883767822
EPSILON_MIN=0
EPSILON_START=1.0
GAMMA=0.985
LEARNING_RATE=0.0003357246838411483
TAU=0.01
FIRST_LAYER_NEURONS=128
SECOND_LAYER_NEURONS=64
MEMORY_CAPACITY=100000
INITIAL_RANDOM_EXPERIENCES=2048
CURRICULUM=True
MAX_TRAIN_STEPS_PER_CURRICULUM=500
NUMBER_OF_CURRICULUMS=3
ANTI_CURRICULUM=True
SAVE_AGENT=True

# Specify a list of seed values
SEEDS=(64)
#0 1 2 3 4 5 6 7 8 9 12 22 2023 420 69 1998 42 13 31 64)

# Loop through the seed values and run the Python script
for SEED in "${SEEDS[@]}"
do
    $PYTHON_EXECUTABLE $SCRIPT_PATH \
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
        --max_train_steps_per_curriculum $MAX_TRAIN_STEPS_PER_CURRICULUM \
        --number_of_curriculums $NUMBER_OF_CURRICULUMS \
        --anti_curriculum "$ANTI_CURRICULUM" \
	--save_agent "$SAVE_AGENT" \
        --seed $SEED
done
