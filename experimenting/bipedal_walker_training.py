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
from models.walker_2d_architectures import critic_network, actor_network, v_network
from rl_utils.replay_buffer import memory_bipedal_walker as memory
from rl_utils.optimization_sac import perform_optimization_step
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Let's check if the environment is working

def train_sac_bipedal_walker(name_env, gpu, alpha, beta, theta, 
                             curriculum, n_episodes, max_t, batch_size, gamma, 
                             tau, grad_clip_value, n_neurons_first_layer, n_neurons_second_layer, buffer_size, project_name,
                             number_of_curriculums, anti_curriculum, seed, sweep, save_agent):
    """
    alpha: learning rate for the actor
    beta: learning rate for the critic
    theta: learning rate for the q network
    n_episodes: number of episodes to train for
    max_t: maximum number of steps per episode
    batch_size: batch size for the replay buffer
    gamma: discount factor
    """
    
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
    actor = actor_network(lr=alpha, n_neurons_first_layer=n_neurons_first_layer, n_neurons_second_layer=n_neurons_second_layer, std_max=2.0).to(device)
    critic_1 = critic_network(lr=beta, n_neurons_first_layer=n_neurons_first_layer, n_neurons_second_layer=n_neurons_second_layer).to(device)
    critic_2 = critic_network(lr=beta, n_neurons_first_layer=n_neurons_first_layer, n_neurons_second_layer=n_neurons_second_layer).to(device)
    v_1 = v_network(lr=theta, n_neurons_first_layer=n_neurons_first_layer, n_neurons_second_layer=n_neurons_second_layer).to(device)
    v_2_target = v_network(lr=theta, n_neurons_first_layer=n_neurons_first_layer, n_neurons_second_layer=n_neurons_second_layer).to(device)

    # Initialize the replay buffer
    replay_buffer = memory(buffer_size=buffer_size)
    
    # Initialize the criterion
    criterion = nn.MSELoss()

    ##### Login to wandb + hyperparameters and metadata
    # Start a new wandb run to track this experiment
    ##### Login wandb + Hyperparameters and metadata
    # Start a new wandb run to track this script
    run = wandb.init(
    # Set the wandb project where this run will be logged
    project=project_name,
    # Track hyperparameters and run metadata
    config={
    "architecture": "Lunar lander basic MLP",
    "environment": name_env,
    "learning_rate_actor": alpha,
    "learning_rate_critic": beta,
    "learning_rate_q": theta,
    "episodes": n_episodes,
    "batch_size": batch_size,
    "gamma": gamma,
    "grad_clip_value": grad_clip_value,
    "n_neurons_first_layer": n_neurons_first_layer,
    "n_neurons_second_layer": n_neurons_second_layer,
    "tau": tau,
    "memory_length": replay_buffer.buffer_size,
    "optimizer": "Adam",
    "device": device,
    "curriculum": curriculum,
    "number_of_curriculums": number_of_curriculums,
    "anti_curriculum": anti_curriculum,
    "seed": seed,})

    run.define_metric("train/step")
    run.define_metric("train_ep/episode")
    run.define_metric("train/*", step_metric="train/step")
    run.define_metric("train_ep/*", step_metric="train_ep/episode")
    run_name = run.name + name_env

    current_episode = 0
    current_step = 0

    ##### End of wandb login
    
    mean_reward_eval_smoothed = deque(maxlen=50)
    mean_reward_eval = 0
    std_reward_eval = 0
    
    # Initialize the training loop
    for episode in tqdm(range(n_episodes)):
        current_state = env.reset()
        done = False
        total_reward = 0
        for i in range(max_t):
            # Select an action
            current_state_tensor = torch.tensor([current_state], device=device, dtype=torch.float32)
            action, _ = actor.sample_action(current_state_tensor, reparameterize=False)
            action = action.detach().cpu().numpy()[0]
            # Perform the action
            next_state, reward, done, _ = env.step(action)
            
            # Add experience transition to replay buffer, but if we are using curriculum learning, 
            # then also calculate temporal difference error and save it as well
            if curriculum:
                # Calculate the temporal difference error
                with torch.no_grad():
                    next_state_tensor = torch.tensor([next_state], device=device, dtype=torch.float32)
                    next_action, next_action_log_prob = actor.sample_action(next_state_tensor)
                    next_action = next_action.detach().cpu().numpy()[0]
                    next_action_log_prob = next_action_log_prob.detach().cpu().numpy()[0]
                    next_state_value = torch.min(v_1(next_state_tensor, next_action), v_2_target(next_state_tensor, next_action))
                    td_error = reward + gamma * next_state_value - v_1(current_state_tensor, torch.tensor([action], device=device, dtype=torch.float32))
                    td_error = td_error.detach().cpu().numpy()[0]
                # Add the experience transition to the replay buffer
                replay_buffer.add(current_state, action, reward, next_state, done, td_error)
            else:
                replay_buffer.add(current_state, action, reward, next_state, done)
                
            # Perform optimization step
            if len(replay_buffer.buffer_deque) > len(minibatch):
                # Get the minibatch from the replay buffer
                minibatch = replay_buffer.sample_batch(batch_size) 
                loss, critic_1_loss, critic_2_loss, v_1_loss, v_2_loss = perform_optimization_step(actor, critic_1, critic_2, v_1, v_2_target, minibatch, 
                                                                                                   gamma, tau, grad_clip_value, device)
                # Logging the losses
                run.log({"train/loss": loss, "train/critic_1_loss": critic_1_loss, "train/critic_2_loss": critic_2_loss, 
                         "train/q_1_loss": v_1_loss, "train/q_2_loss": v_2_loss})

            # Update the current state
            current_state = next_state.copy()
            total_reward += reward
            current_step += 1

            ###### Logging to wandb
            # Evaluate the agent every 500 steps
            if current_step % 500 == 0:
                env_eval.reset(seed=42)
                rewards = []
                for _ in range(20):
                    current_state_eval = env_eval.reset()[0]
                    done_eval = False
                    reward_acc = 0
                    for k in range(max_t):
                        action_eval, _ = actor.sample_action(torch.tensor([current_state_eval], device=device, dtype=torch.float32), reparameterize=False)
                        next_state_eval, reward, done_eval, _, info = env_eval.step(action_eval)
                        current_state_eval = next_state_eval
                        reward_acc += reward
                        if done_eval:
                            break
                    rewards.append(reward_acc)
                mean_reward_eval = sum(rewards)/len(rewards)
                std_reward_eval = np.std(rewards).item()
                mean_reward_eval_smoothed.append(mean_reward_eval)
                run.log({"train/mean_reward_eval": mean_reward_eval, "train/mean_reward_eval_smoothed": sum(mean_reward_eval_smoothed)/len(mean_reward_eval_smoothed), 
                         "train/std_reward_eval": std_reward_eval})
                actor.save_model(episode=episode, train_step=current_step, optimizer=actor.optimizer, loss=loss, buffer=replay_buffer, path='experimenting/models/trained_models/checkpoint_bipedal')

            # Calculating the gradients of the networks
            squared_norm_gradients_actor = 0
            for w in actor.parameters():
                squared_norm_gradients_actor += (torch.norm(w.grad)**2).item()
            squared_norm_gradients_critic = 0
            for w in critic_1.parameters():
                squared_norm_gradients_critic += (torch.norm(w.grad)**2).item()
            squared_norm_gradients_value = 0
            for w in v_1.parameters():
                squared_norm_gradients_value += (torch.norm(w.grad)**2).item()

            # Logging metrics to wandb
            log_dict = {
                "train/step": current_step,
                "train/training_reward": reward,
                "train/training_episode": episode,
                "train/loss": loss,
                "train/squared_norm_gradients_actor": squared_norm_gradients_actor,
                "train/squared_norm_gradients_critic": squared_norm_gradients_critic,
                "train/squared_norm_gradients_value": squared_norm_gradients_value,
                "train/action_taken": action,
                "train/replay_buffer_#ofexperiences": len(replay_buffer.buffer_deque)
            }
            if curriculum:
                log_dict["train/curriculum"] = replay_buffer.curriculum
            run.log(log_dict)

            # For lunar lander, it is considered solved after 200 points
            # This should be only available for sweeping, as we want results as fast as possible
            if mean_reward_eval > 200 and sweep:
                run.save
                run.finish()
                if save_agent:
                    torch.save(actor.state_dict(), 'experimenting/models/trained_models/' + run_name + '.pt')
                return
            ###### End of logging to wandb


            # Check if the episode is done
            if done:
                break
        
        ## Outer loop logging
        # Logging the total reward of the training episode
        log_dict = {
            "train_ep/episode_reward": total_reward,
            "train_ep/episode": episode,
            "train_ep/mean_reward_eval": mean_reward_eval,
            "train_ep/std_reward_eval": std_reward_eval
        }
        run.log(log_dict)

    # Post-training code
    run.save
    run.finish()
    if save_agent:
        torch.save(actor.state_dict(), 'experimenting/models/trained_models/lunar_lander/' + run_name + '.pt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for training a SAC agent on the BipedalWalker environment')
    # Misc arguments
    parser.add_argument('--name_env', type=str, default="BipedalWalker-v3", help='Name of the environment')
    parser.add_argument('-gpu', '--gpu', metavar='GPU', type=str, help='gpu to use', default='3') # Only 3 and 4 should be used. Number 2 could also be used but check availability first
    