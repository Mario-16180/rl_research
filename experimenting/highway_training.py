import gymnasium as gym
import highway_env
import torch
import torch.nn as nn
import pickle
import wandb
import warnings
import argparse
import numpy as np
import random
from tqdm import tqdm
from collections import deque
from models.lunar_lander_dqn_architecture import lunar_lander_mlp
from rl_utils.replay_buffer import memory_lunar_lander as memory
from rl_utils.replay_buffer import memory_lunar_lander_curriculum as memory_with_curriculum
from experimenting.rl_utils.optimization_dqn import perform_optimization_step

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import matplotlib.pyplot as plt

# Let's see first if the highway environment is available

if __name__ == "__main__":
    env = gym.make('highway-fast-v0', render_mode='rgb_array')
    env.configure({"action": {"type": "DiscreteMetaAction"},
                   "observation": {"type": "GrayscaleObservation",
                                   "weights": [0.2989, 0.5870, 0.1140],
                                   "stack_size": 4,
                                   "observation_shape": (128, 64),
                                   "scaling": 3},
                   "lanes_count": 4,
                   "vehicles_count": 50,
                   "duration": 1,
                   "collision_reward": -1,
                   "show_trajectories": False,
                   })
    #env.configure({"manual_control": True,
    #               "duration": 5,})
    env.reset()
    done = False
    rewards = []
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, done, info, a = env.step(action)
        rewards.append(reward)
        # print the stacked observations in grayscale

        #env.render()
    plt.imshow(obs[0, :, :])
    plt.show()
    plt.imshow(obs[1, :, :])
    plt.show()
    plt.imshow(obs[2, :, :])
    plt.show()
    plt.imshow(obs[3, :, :])
    plt.show()
    print(np.sum(rewards), len(rewards))
    #env.close()
