#!/bin/bash

# Set the path to your Python executable
PYTHON_EXECUTABLE=python

# Specify your Python script and its arguments
SCRIPT_PATH=experimenting/lunar_lander_training.py
EPISODES=500
GRAD_CLIP_VALUE=384
EPSILON_DECAY=0.012757271665826056
LEARNING_RATE=0.0014387908659690462
TAU=0.0043900818020396705
FIRST_LAYER_NEURONS=256
SECOND_LAYER_NEURONS=128
INITIAL_RANDOM_EXPERIENCES=35000
CURRICULUM=True
SAVE_AGENT=True
MAX_TRAIN_STEPS_PER_CURRICULUM_CRITERION1=2000
NUMBER_OF_CURRICULUMS=8
ANTI_CURRICULUM=True
PERCENTILE=0

# Specify a list of seed values
SEEDS=(0 1 2 3 4 5 6 7 8 9 12 22 2023 420 69 1998 42 13 31 64)

# Loop through the seed values and run the Python script
for SEED in "${SEEDS[@]}"
do
    $PYTHON_EXECUTABLE $SCRIPT_PATH \
        --episodes $EPISODES \
        --grad_clip_value $GRAD_CLIP_VALUE \
        --epsilon_decay $EPSILON_DECAY \
        --learning_rate $LEARNING_RATE \
        --tau $TAU \
        --first_layer_neurons $FIRST_LAYER_NEURONS \
        --second_layer_neurons $SECOND_LAYER_NEURONS \
        --initial_random_experiences $INITIAL_RANDOM_EXPERIENCES \
        --curriculum "$CURRICULUM" \
	--save_agent "$SAVE_AGENT" \
        --max_train_steps_per_curriculum_criterion1 $MAX_TRAIN_STEPS_PER_CURRICULUM_CRITERION1 \
	--number_of_curriculums $NUMBER_OF_CURRICULUMS \
	--anti_curriculum "$ANTI_CURRICULUM" \
	--percentile $PERCENTILE \
	--seed $SEED
done
