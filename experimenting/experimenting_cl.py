import torch
import torch.nn as nn
from models.impala_cnn_architecture import impala_cnn
from matplotlib import pyplot as plt
import gym
from rl_utils.replay_buffer import memory_with_curriculum

model = impala_cnn()
env = gym.make('procgen:procgen-bossfight-v0', num_levels=0, start_level=0)
observation = env.reset()

memory_object = memory_with_curriculum(max_size=10000)
memory_object.populate_memory_model(env, k_initial_experiences=1000)
print(len(memory_object.buffer_deque))
