import gym
import torch
import torch.nn as nn
import wandb
import pandas as pd
import warnings
import numpy as np
import argparse
from tqdm import tqdm
from models.impala_cnn_architecture import impala_cnn
from experimenting.rl_utils.stack_frames import stacked_frames_class
from experimenting.rl_utils.replay_buffer import memory
from experimenting.rl_utils.replay_buffer import memory_with_curriculum
from experimenting.rl_utils.optimization import perform_optimization_step
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def train_dqn_curriculum(name_env, episodes, batch_size, gamma, epsilon_start, epsilon_decay, epsilon_min,  
                      optimizer=None, criterion=None, learning_rate=0.0001, tau=0.001, model=None, replay_buffer=None, 
                      num_levels=500, num_levels_eval=20, start_level=0, start_level_test=1024, background=False,
                      initial_random_experiences=5000, memory_capacity=50000, resume=False, project_name="rl_research_mbzuai",
                      number_of_curriculums=3, curriculum=False):
    # Used to normalize the reward
    rewardbounds_per_env=pd.read_csv('experimenting/rl_utils/reward_data_per_environment.csv', delimiter=' ', header=0)
    min_r = rewardbounds_per_env[rewardbounds_per_env.Environment == name_env].Rminhard.item()
    max_r = rewardbounds_per_env[rewardbounds_per_env.Environment == name_env].Rmaxhard.item()
    normalize_reward = lambda r: (r - min_r) / (max_r - min_r)

    # Initialize environment for training and evaluation
    env = gym.make(name_env, start_level=start_level, num_levels=num_levels, use_backgrounds=background)
    env_eval = gym.make(name_env, start_level=start_level_test, num_levels=num_levels_eval, use_backgrounds=background)

    # Choose GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model and target model
    if model is None:
        model_policy = impala_cnn(env).to(device)
        model_target = impala_cnn(env).to(device)
        model_target.load_state_dict(model_policy.state_dict())

    # Initialize replay buffer if it's empty
    if replay_buffer is None:
        if curriculum:
            replay_buffer = memory_with_curriculum(max_size=memory_capacity, curriculums=number_of_curriculums)
            replay_buffer.populate_memory_model(model_policy, env, name_env, k_initial_experiences=initial_random_experiences, device=device)
        else:
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
    "background": background,
    "number_of_curriculums": number_of_curriculums,},
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
            norm_reward = normalize_reward(reward)
            current_state = stacked_frames.stacked_frames_array
            stacked_frames.append_frame_to_stack(next_obs)
            next_state = stacked_frames.stacked_frames_array
            
            minibatch = replay_buffer.sample(batch_size)

            loss = perform_optimization_step(model_policy, model_target, minibatch, gamma, optimizer, criterion, device, batch_size, curriculum=True)

            if curriculum: 
                # Calculate temporal difference
                with torch.no_grad():
                    next_state = torch.tensor([next_state], device=device, dtype=torch.float32)
                    current_state = torch.tensor([current_state], device=device, dtype=torch.float32)
                    temporal_difference = ((norm_reward + gamma * model_policy.forward(next_state).max(1)[0].item() - 
                                            model_policy.forward(current_state)[0,action])**2).item()
                replay_buffer.add((current_state, action, norm_reward, next_state, done, temporal_difference))
            else:
                replay_buffer.add((current_state, action, norm_reward, next_state, done))
            
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
                model_policy.save_model(episode=episode, optimizer=optimizer, loss=loss, buffer=replay_buffer, path='experimenting/models/trained_models/checkpoint.tar')

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
            if curriculum:
                log_dict["train/curriculum"] = replay_buffer.curriculum
            run.log(log_dict)
        scheduler.step()
    run.save
    run.finish()
    return model_policy, replay_buffer.buffer_deque, run_name

if __name__ == '__main__':

    (name_env, episodes, batch_size, gamma, epsilon_start, epsilon_decay, epsilon_min,  
                      optimizer=None, criterion=None, learning_rate=0.0001, tau=0.001, model=None, replay_buffer=None, 
                      num_levels=500, num_levels_eval=20, start_level=0, start_level_test=1024, background=False,
                      initial_random_experiences=5000, memory_capacity=50000, resume=False, project_name="rl_research_mbzuai",
                      number_of_curriculums=3, curriculum=False):
    parser = argparse.ArgumentParser(description='Arguments for training the DQN agent')
    parser.add_argument('name_env', metavar='N', type=str, nargs='+',)
    parser.add_argument('episodes', metavar='E', type=int, nargs='+',)
    env_name = "procgen:procgen-bossfight-v0"
    learned_model, replay_buffer, run_name = train_dqn_curriculum(env_name, episodes=5000, batch_size=64, gamma=0.99, 
                                                        epsilon_start=0.99, epsilon_decay=150000, epsilon_min=0.05, learning_rate=0.001, 
                                                        num_levels=500, num_levels_eval=20, background=False, start_level=0, start_level_test=42,
                                                        resume=False)