import imageio
import os
import gym
import torch
import torch.nn as nn
import wandb
import random
import math
import pandas as pd
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

def perform_optimization_step(model_policy, model_target, minibatch, gamma, optimizer, criterion, device, batch_size):
    state_batch = torch.cat([torch.tensor([s], device=device, dtype=torch.float32) for (s, a, r, s_, d) in minibatch])
    action_batch = torch.cat([torch.tensor([a], device=device, dtype=torch.int64) for (s, a, r, s_, d) in minibatch]).reshape([batch_size,1])
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
    torch.nn.utils.clip_grad_norm_(model_policy.parameters(), 100) 
    optimizer.step()
    
    return loss

def train_new_cl_algo():
    pass

def train_dqn(name_env, episodes, batch_size, gamma, epsilon_start, epsilon_decay, epsilon_min,  
                      optimizer=None, criterion=None, learning_rate=0.0001, tau=0.001, model=None, replay_buffer=None, 
                      num_levels=500, num_levels_eval=20, start_level=0, start_level_test=1024, background=False,
                      initial_random_experiences=5000, memory_capacity=50000, resume=False, project_name="rl_research_mbzuai"):
    rewardbounds_per_env=pd.read_csv('rl_utils/reward_data_per_environment.csv', delimiter=' ', header=0)
    min_r = rewardbounds_per_env[rewardbounds_per_env.Environment == name_env].Rminhard.item()
    max_r = rewardbounds_per_env[rewardbounds_per_env.Environment == name_env].Rmaxhard.item()
    normalize_reward = lambda r: (r - min_r) / (max_r - min_r)
    env = gym.make(name_env, start_level=start_level, num_levels=num_levels, use_backgrounds=background)
    env_eval = gym.make(name_env, start_level=start_level_test, num_levels=num_levels_eval, use_backgrounds=background)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize model and target model
    if model is None:
        model_policy = impala_cnn(env).to(device)
        model_target = impala_cnn(env).to(device)
        model_target.load_state_dict(model_policy.state_dict())
    # Initialize replay buffer if it's empty
    if replay_buffer is None:
        replay_buffer = memory(max_size=memory_capacity)
        replay_buffer.populate_memory_random(env, name_env, k_initial_experiences=initial_random_experiences)
    if optimizer is None:
        optimizer = torch.optim.Adam(model_policy.parameters(), lr=learning_rate)
    if criterion is None:
        criterion = nn.MSELoss()
    current_episode = 0
    ##### Login wandb + Hyperparameters and metadata
    # Start a new wandb run to track this script
    run = wandb.init(
    # Set the wandb project where this run will be logged
    project=project_name,
    # Track hyperparameters and run metadata
    config={
    "architecture": "IMPALA_large_CNN",
    "dataset": name_env,
    "learning_rate": learning_rate,
    "episodes": episodes,
    "batch_size": batch_size,
    "gamma": gamma,
    "epsilon_start": epsilon_start,
    "epsilon_decay": epsilon_decay,
    "epsilon_min": epsilon_min,
    "tau": tau,
    "memory_length": replay_buffer.buffer_size,
    "optimizer": optimizer,
    "criterion": criterion,
    "device": device,
    "num_levels": num_levels,
    "num_levels_eval": num_levels_eval,
    "background": background,},
    resume=resume,
    )
    run.define_metric("train/step")
    run.define_metric("train/*", step_metric="train/step")
    run_name = run.name
    if wandb.run.resumed:
        checkpoint = torch.load(wandb.restore('models/trained_models/checkpoint.tar'))
        model_policy = impala_cnn(env).to(device)
        model_policy.load_state_dict(checkpoint['model_state_dict'])
        model_target = impala_cnn(env).to(device)
        model_target.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        current_episode = checkpoint['episode']
        loss = checkpoint['loss']
        replay_buffer = checkpoint['buffer']
    ##### End of wandb login

    current_step = 0
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
    for episode in tqdm(range(current_episode, episodes)):
        obs = env.reset()
        stacked_frames = stacked_frames_class()
        stacked_frames.initialize_stack(obs)
        done = False
        train_reward = 0
        while not done:
            action, eps = model_policy.select_action(env, stacked_frames.stacked_frames_array, epsilon_start, epsilon_decay, epsilon_min, current_step, device)
            action = action.item()
            next_obs, reward, done, _ = env.step(action)
            reward = normalize_reward(reward)
            current_state = stacked_frames.stacked_frames_array
            stacked_frames.append_frame_to_stack(next_obs)
            next_state = stacked_frames.stacked_frames_array
            
            replay_buffer.add((current_state, action, reward, next_state, done))

            
            minibatch = replay_buffer.sample(batch_size)

            loss = perform_optimization_step(model_policy, model_target, minibatch, gamma, optimizer, criterion, device, batch_size)

            # Update target network
            policy_weights = model_policy.state_dict()
            target_weights = model_target.state_dict()
            for name in policy_weights:
                target_weights[name] = tau * policy_weights[name] + (1 - tau) * target_weights[name]

            # Counters
            current_step += 1
            train_reward += reward

            if current_step % 250 == 0:
                for eval_episode in range(20):
                    obs = env_eval.reset()
                    stacked_frames_test = stacked_frames_class()
                    stacked_frames_test.initialize_stack(obs)
                    done = False
                    reward_acc = 0
                    while not done:
                        action_eval, _ = model_policy.select_action(env_eval, stacked_frames_test.stacked_frames_array, 0.05, 1, 0.05, current_step, device)
                        action_eval = action_eval.item()
                        next_obs, reward, done, _ = env_eval.step(action_eval)
                        stacked_frames_test.append_frame_to_stack(next_obs)
                        reward_acc += reward
                        if done:
                            break
                    run.log({f"train/reward_eval_{eval_episode}": reward_acc})
                model_policy.save_model(episode=episode, optimizer=optimizer, loss=loss, buffer=replay_buffer, path='models/trained_models/checkpoint.tar')

            # Logging metrics to wandb
            squared_norm_gradients = 0
            for w in model_policy.parameters():
                squared_norm_gradients += torch.norm(w.grad)**2
            log_dict = {
                "train/step": current_step,
                "train/training_reward_normalized": train_reward,
                "train/training_episode": episode,
                "train/loss": loss,
                "train/squared_norm_gradients": squared_norm_gradients,
                "train/action_taken": action,
                "train/epsilon": eps,
                "train/replay_buffer_#ofexperiences": len(replay_buffer.buffer_deque),
                "train/learning_rate": scheduler.get_last_lr()[0],
            }
            run.log(log_dict)
        scheduler.step()
    run.save
    run.finish()
    return model_policy, replay_buffer, run_name

if __name__ == '__main__':
    env_name = "procgen:procgen-bossfight-v0"
    learned_model, replay_buffer, run_name = train_dqn(env_name, episodes=5000, batch_size=64, gamma=0.99, 
                                                        epsilon_start=0.99, epsilon_decay=150000, epsilon_min=0.05, learning_rate=0.001, 
                                                        num_levels=500, num_levels_eval=20, background=False, start_level=0, start_level_test=42,
                                                        resume=True)