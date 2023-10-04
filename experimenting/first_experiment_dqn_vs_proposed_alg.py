import imageio
import os
import gym
import torch
import torch.nn as nn
import wandb
from models.impala_cnn_architecture import impala_cnn
from rl_utils.stack_frames import stacked_frames_class

def save_frames_as_gif(frames, path='/home/mario.cantero/Documents/Research/rl_research/SavedRenders', filename='gym_animation.gif'):
    imageio.mimwrite(os.path.join(path, filename), frames, duration=20, format='gif')

def train_new_cl_algo():
    pass

def train_vanilla_dqn(env, model):
    pass

if __name__ == '__main__':
    # start a new wandb run to track this script
    """
    wandb.init(
    # set the wandb project where this run will be logged
    project="rl_research_mbzuai",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.02,
    "architecture": "IMPALA_large_CNN",
    "dataset": "procgen environment",
    "epochs": 1000}
    )
    """
    env_name = "procgen:procgen-bossfight-v0"
    env = gym.make(env_name, num_levels=1, start_level=0)
    obs = env.reset()

    q_function = impala_cnn(env)
    stacked_frames = stacked_frames_class()
    stacked_frames.initialize_stack(obs)
    print(q_function(stacked_frames.stacked_frames_array))

    """
    while True:
        obs, rew, done, info = env.step(env.action_space.sample())
        print(rew, 'step: ', iter)
        iter += 1
        #frame = env.render()
        frames.append(obs)
        if done:
            break
    save_frames_as_gif(frames, filename='pruebita2.gif')
    """