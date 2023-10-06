import imageio
import os
import gym
import torch
import torch.nn as nn
import wandb
import random
import math
import warnings
import numpy as np
from tqdm import tqdm
from models.impala_cnn_architecture import impala_cnn
from rl_utils.stack_frames import stacked_frames_class
from rl_utils.replay_buffer import memory

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def save_frames_as_gif(frames, path='/home/mario.cantero/Documents/Research/rl_research/SavedRenders', filename='gym_animation.gif'):
    imageio.mimwrite(os.path.join(path, filename), frames, duration=20, format='gif')

def select_action(model, env, state, epsilon_start, epsilon_decay, epsilon_min, current_step, device):
    sample_for_probability = random.random()
    eps_threshold = epsilon_min + (epsilon_start - epsilon_min) * math.exp(-1. * current_step / epsilon_decay)
    if sample_for_probability > eps_threshold:
        with torch.no_grad():
            state = torch.tensor([state], device=device, dtype=torch.float32)
            action = model(state).max(1)[1].view(1, 1)
            return action
    else:
        action = torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
        return action

def perform_optimization_step(model_policy, model_target, minibatch, gamma, optimizer, criterion, device):
    state_batch = torch.cat([torch.tensor([s], device=device, dtype=torch.float32) for (s, a, r, s_, d) in minibatch])
    action_batch = torch.cat([torch.tensor([a], device=device, dtype=torch.int64) for (s, a, r, s_, d) in minibatch]).reshape([32,1])
    reward_batch = torch.cat([torch.tensor([r], device=device, dtype=torch.float32) for (s, a, r, s_, d) in minibatch])
    next_state_batch = torch.cat([torch.tensor([s_], device=device, dtype=torch.float32) for (s, a, r, s_, d) in minibatch])
    done_batch = torch.cat([torch.tensor([d], device=device, dtype=torch.bool) for (s, a, r, s_, d) in minibatch])

    state_action_q_values_policy = model_policy(state_batch).gather(1, action_batch)
    
    with torch.no_grad():
        state_action_q_values_target = model_target(next_state_batch).max(1)[0].detach()

    expected_state_action_q_values = reward_batch + (1 - done_batch * 1) * (state_action_q_values_target * gamma)

    loss = criterion(state_action_q_values_policy, expected_state_action_q_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # torch.nn.utils.clip_grad_norm_(model_policy.parameters(), 0.5)
    return loss

def train_new_cl_algo():
    pass

def train_vanilla_dqn(name_env, episodes, max_steps, batch_size, gamma, epsilon_start, epsilon_decay, epsilon_min,  
                      optimizer=None, criterion=None, learning_rate=0.0001, tau=0.01, model=None, replay_buffer=None, 
                      num_levels=0, num_levels_eval=20):
    env = gym.make(name_env, start_level=0, num_levels=num_levels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize model and target model
    if model is None:
        model_policy = impala_cnn(env).to(device)
        model_target = impala_cnn(env).to(device)
        model_target.load_state_dict(model_policy.state_dict())
    else:
        model_policy = model.to(device)
        model_target = impala_cnn(env).to(device)
        model_target.load_state_dict(model_policy.state_dict())

    # Initialize replay buffer if it's empty
    if replay_buffer is None:
        replay_buffer = memory(max_size=200000)
        replay_buffer.populate_memory_random(env, k_initial_experiences=20000)

    if optimizer is None:
        optimizer = torch.optim.Adam(model_policy.parameters(), lr=learning_rate)
    if criterion is None:
        criterion = nn.MSELoss()
    
    ##### Login wandb + Hyperparameters and metadata
    # Start a new wandb run to track this script
    run = wandb.init(
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
    run.define_metric("train/step")
    run.define_metric("train/*", step_metric="train/step")
    run_name = run.name
    ##### End of wandb login

    current_step = 0
    for episode in tqdm(range(episodes)):
        obs = env.reset()
        stacked_frames = stacked_frames_class()
        stacked_frames.initialize_stack(obs)

        done = False
        while not done:
            action = select_action(model_policy, env, stacked_frames.stacked_frames_array, epsilon_start, epsilon_decay, epsilon_min, current_step, device)
            action = action.item()
            next_obs, reward, done, _ = env.step(action)
            
            current_state = stacked_frames.stacked_frames_array
            stacked_frames.append_frame_to_stack(next_obs)
            next_state = stacked_frames.stacked_frames_array
            
            replay_buffer.add((current_state, action, reward, next_state, done))

            minibatch = replay_buffer.sample(batch_size)

            loss = perform_optimization_step(model_policy, model_target, minibatch, gamma, optimizer, criterion, device)

            policy_weights = model_policy.state_dict()
            target_weights = model_target.state_dict()
            for name in policy_weights:
                target_weights[name] = tau * policy_weights[name] + (1 - tau) * target_weights[name]

            current_step += 1

            if current_step % 10000 == 0:
                squared_norm_gradients = 0
                for w in model_policy.parameters():
                    squared_norm_gradients += torch.norm(w)**2
                reward_test = []
                for _ in range(20):
                    env = gym.make(name_env, start_level=0, num_levels=num_levels_eval)
                    obs = env.reset()
                    stacked_frames_test = stacked_frames_class()
                    stacked_frames_test.initialize_stack(obs)
                    done = False
                    reward_acc = 0
                    while not done:
                        action = select_action(model_policy, env, stacked_frames_test.stacked_frames_array, 0.05, 1, 0.05, current_step, device)
                        action = action.item()
                        next_obs, reward, done, _ = env.step(action)
                        stacked_frames_test.append_frame_to_stack(next_obs)
                        if done:
                            break
                        reward_acc += reward
                    reward_test.append(reward_acc)
                env = gym.make(name_env, start_level=0, num_levels=num_levels)
                # Logging metrics to wandb
                log_dict = {
                    "train/step": current_step,
                    "train/episode": episode,
                    "train/reward": np.mean(reward_test),
                    "train/reward_std": np.std(reward_test),
                    "train/loss": loss,
                    "train/squared_norm_gradients": squared_norm_gradients,
                }
                run.log(log_dict)
    
    run.finish()
    return model_policy, replay_buffer, run_name

if __name__ == '__main__':
    env_name = "procgen:procgen-bossfight-v0"
    learned_model, replay_buffer, run_name = train_vanilla_dqn(env_name, episodes=10000, max_steps=1000, batch_size=32, gamma=0.99, 
                                                     epsilon_start=0.99, epsilon_decay=100000, epsilon_min=0.05, learning_rate=0.001, 
                                                     num_levels=1, num_levels_eval=1)
    learned_model.save_model(path=f"models/trained_models/impala_cnn_{run_name}.pt")