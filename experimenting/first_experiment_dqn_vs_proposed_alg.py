import imageio
import os
import gym
import torch
import torch.nn as nn
import wandb
import random
import math
from models.impala_cnn_architecture import impala_cnn
from rl_utils.stack_frames import stacked_frames_class
from rl_utils.replay_buffer import memory

def save_frames_as_gif(frames, path='/home/mario.cantero/Documents/Research/rl_research/SavedRenders', filename='gym_animation.gif'):
    imageio.mimwrite(os.path.join(path, filename), frames, duration=20, format='gif')

def select_action(model, env, state, epsilon_start, epsilon_decay, epsilon_min, current_step, device):
    sample_for_probability = random.random()
    eps_threshold = epsilon_min + (epsilon_start - epsilon_min) * math.exp(-1. * current_step / epsilon_decay)
    if sample_for_probability > eps_threshold:
        with torch.no_grad():
            action = model(state).max(1)[1].view(1, 1)
            return action
    else:
        action = torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
        return action

def perform_optimization_step(model, minibatch, gamma, optimizer, criterion, device):
    pass

def train_new_cl_algo():
    pass

def train_vanilla_dqn(model, name_env, episodes, max_steps, batch_size, gamma, epsilon_start, epsilon_decay, epsilon_min, tau, 
                      replay_buffer, optimizer, criterion, learning_rate):
    ##### Login wandb + Hyperparameters and metadata
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Start a new wandb run to track this script
    wandb.init(
    # Set the wandb project where this run will be logged
    project="rl_research_mbzuai",
    # Track hyperparameters and run metadata
    config={
    "architecture": "IMPALA_large_CNN",
    "dataset": name_env,
    "learning_rate": learning_rate,
    "episodes": episodes,
    "max_steps": max_steps,
    "batch_size": batch_size,
    "gamma": gamma,
    "epsilon_start": epsilon_start,
    "epsilon_decay": epsilon_decay,
    "epsilon_min": epsilon_min,
    "tau": tau,
    "memory_len": replay_buffer.buffer_size,
    "optimizer": optimizer,
    "criterion": criterion,
    "device": device,},
    )

    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    
    current_step = 0
    buffer = memory()
    for i in range(episodes):
        env = gym.make(name_env, start_level=0)
        obs = env.reset()
        stacked_frames = stacked_frames_class()
        stacked_frames.initialize_stack(obs)

        done = False
        while not done:
            action = select_action(model, stacked_frames.stacked_frames_array, epsilon_start, epsilon_decay, epsilon_min, current_step, device)
            next_obs, reward, done, _ = env.step(action)
            stacked_frames.append_frame_to_stack(next_obs)
            memory
            
            
            log_dict = {
                "train/step": t,
                "train/episode": i,
                "train": i,
            }
            wandb.log(log_dict)
    
    wandb.finish()

if __name__ == '__main__':
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