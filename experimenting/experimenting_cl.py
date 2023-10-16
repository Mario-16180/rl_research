import torch
import torch.nn as nn
from models.impala_cnn_architecture import impala_cnn
from matplotlib import pyplot as plt
import gym
from rl_utils.replay_buffer import memory_with_curriculum
import os
import random

# Print current working directory
os.chdir('experimenting')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
name_env = 'procgen:procgen-bossfight-v0'
env = gym.make(name_env, num_levels=0, start_level=0)
observation = env.reset()
model = impala_cnn(env).to(device)
memory_object = memory_with_curriculum(max_size=50000)
memory_object.populate_memory_model(model, env, name_env, k_initial_experiences=10000, device=device)
td_list = [memory_object.buffer_deque[i][-1] for i in range(len(memory_object.buffer_deque))]
# Erase 90 % of the experiences that fall between the 25th and 75th quantiles
td_list_undersampled = [td_list[i] for i in range(len(td_list)) if ((random.random() < 0.1) and 
                        (td_list[i] > torch.quantile(torch.tensor(td_list), 0.10)) and (td_list[i] < torch.quantile(torch.tensor(td_list), 0.90))) 
                        or td_list[i] <= torch.quantile(torch.tensor(td_list), 0.10) or td_list[i] >= torch.quantile(torch.tensor(td_list), 0.90)]

print(f"Lenth: {len(td_list_undersampled)}, max td: {max(td_list_undersampled)}, min td: {min(td_list_undersampled)}")
# Plot the quantiles
print(f"Quantiles: {torch.quantile(torch.tensor(td_list_undersampled), torch.tensor([0.25, 0.5, 0.75]))}")
# Plot the distribution with a line for the quantiles
plt.axvline(torch.quantile(torch.tensor(td_list_undersampled), 0.25), color='r')
plt.axvline(torch.quantile(torch.tensor(td_list_undersampled), 0.5), color='r')
plt.axvline(torch.quantile(torch.tensor(td_list_undersampled), 0.75), color='r')
# Plot the distribution
plt.hist(td_list_undersampled, bins=150)
plt.show()

# Check the distr
print(len(memory_object.buffer_deque))
