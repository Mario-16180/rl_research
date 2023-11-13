import gym
import torch
import torch.nn as nn
import pickle
import wandb
import pandas as pd
import warnings
import argparse
from tqdm import tqdm
from collections import deque
from models.impala_cnn_architecture import impala_cnn
from rl_utils.stack_frames import stacked_frames_class
from rl_utils.replay_buffer import memory
from rl_utils.replay_buffer import memory_with_curriculum
from rl_utils.optimization import perform_optimization_step
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# This script is used to do a fast training in the LunarLander-v2 environment
if __name__ == '__main__':
    