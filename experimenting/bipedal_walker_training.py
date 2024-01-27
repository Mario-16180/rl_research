import gym
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
from models.walker_2d_architectures import critic_network, actor_network, q_network
from rl_utils.replay_buffer import memory_lunar_lander as memory
from rl_utils.replay_buffer import memory_lunar_lander_curriculum as memory_with_curriculum
from rl_utils.optimization import perform_optimization_step
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Let's check if the environment is working

def train_sac_bipedal_walker(name_env, gpu):
    # Initialize the environment
    env = gym.make(name_env)
    env_eval = gym.make(name_env)

    # Choose GPU if available
    if torch.cuda.device_count() > 1:
        device = torch.device("cuda:"+gpu if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

    # Initialize the networks
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for training a SAC agent on the BipedalWalker environment')
    # Misc arguments
    parser.add_argument('--name_env', type=str, default="BipedalWalker-v3", help='Name of the environment')
    parser.add_argument('-gpu', '--gpu', metavar='GPU', type=str, help='gpu to use', default='3') # Only 3 and 4 should be used. Number 2 could also be used but check availability first
    