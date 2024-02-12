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
from models.walker_2d_architectures import critic_network, actor_network, temperature_factor_updater
from rl_utils.replay_buffer import memory_bipedal_walker as memory
from rl_utils.optimization_sac import perform_optimization_step
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Let's check if the environment is working

def train_sac_bipedal_walker(name_env, gpu, alpha, beta, theta, use_curriculum, max_frame_steps, max_t, batch_size, start_steps, functions_updates, gamma, 
                             tau, grad_clip_value, n_neurons_first_layer, n_neurons_second_layer, log_std_max, buffer_size, project_name,
                             number_of_curriculums, anti_curriculum, seed, save_agent):
    """
    alpha: learning rate for the actor
    beta: learning rate for the critic
    theta: learning rate for the alpha factor
    max_frame_steps: maximum number of frame steps
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
    actor = actor_network(lr=alpha, n_neurons_first_layer=n_neurons_first_layer, n_neurons_second_layer=n_neurons_second_layer, device=device, log_std_max=log_std_max)
    critic_1 = critic_network(lr=beta, n_neurons_first_layer=n_neurons_first_layer, n_neurons_second_layer=n_neurons_second_layer, device=device)
    critic_2 = critic_network(lr=beta, n_neurons_first_layer=n_neurons_first_layer, n_neurons_second_layer=n_neurons_second_layer, device=device)
    critic_1_target = critic_network(lr=beta, n_neurons_first_layer=n_neurons_first_layer, n_neurons_second_layer=n_neurons_second_layer, device=device)
    critic_2_target = critic_network(lr=beta, n_neurons_first_layer=n_neurons_first_layer, n_neurons_second_layer=n_neurons_second_layer, device=device)
    critic_1_target.load_state_dict(critic_1.state_dict())
    critic_2_target.load_state_dict(critic_2.state_dict())
    alpha_factor = temperature_factor_updater(lr=theta, device=device)

    # Initialize the replay buffer
    replay_buffer = memory(buffer_size=buffer_size)

    # Auxiliar function to save the model and log final metrics
    def save_model_and_log_final_metrics():
        # Post-training code
        # Logging the total reward of the training episode
        log_dict = {
            "train_ep/episode_reward": total_reward,
            "train_ep/episode": current_episode,
            "train_ep/mean_reward_eval": mean_reward_eval,
            "train_ep/mean_reward_eval_smoothed": average_reward_eval,
            "train_ep/std_reward_eval": std_reward_eval
        }
        run.log(log_dict)
        run.save
        run.finish()
        if save_agent:
            torch.save(actor.state_dict(), 'experimenting/models/trained_models/bipedal_walker/' + run_name + '.pt')

    ##### Login to wandb + hyperparameters and metadata
    # Start a new wandb run to track this experiment
    ##### Login wandb + Hyperparameters and metadata
    # Start a new wandb run to track this script
    run = wandb.init(
    # Set the wandb project where this run will be logged
    project=project_name,
    # Track hyperparameters and run metadata
    config={
    "architecture": "Bipedal Walker Basic MLP",
    "environment": name_env,
    "learning_rate_actor": alpha,
    "learning_rate_critic": beta,
    "learning_rate_q": theta,
    "max_frame_steps": max_frame_steps,
    "batch_size": batch_size,
    "start_steps": start_steps,
    "gamma": gamma,
    "log_std_max": log_std_max,
    "grad_clip_value": grad_clip_value,
    "n_neurons_first_layer": n_neurons_first_layer,
    "n_neurons_second_layer": n_neurons_second_layer,
    "initial_alpha": alpha_factor.alpha,
    "tau": tau,
    "memory_length": replay_buffer.buffer_size,
    "optimizer": "Adam",
    "device": device,
    "use_curriculum": use_curriculum,
    "number_of_curriculums": number_of_curriculums,
    "anti_curriculum": anti_curriculum,
    "seed": seed,})
    run.define_metric("train/step")
    run.define_metric("train_ep/episode")
    run.define_metric("train/*", step_metric="train/step")
    run.define_metric("train_ep/*", step_metric="train_ep/episode")
    run_name = run.name + name_env

    current_episode = 0
    current_frame_step = 0
    ##### End of wandb login
    mean_reward_eval_smoothed = deque(maxlen=100)
    mean_reward_eval = 0
    average_reward_eval = 0
    std_reward_eval = 0
    actor_loss = 0
    critic_loss_1 = 0
    critic_loss_2 = 0
    alpha_loss = 0
    alpha_value = 0
    q1_mean = 0
    q2_mean = 0
    current_optimization_steps = 0
    progress_bar = iter(tqdm(range(max_frame_steps)))
    # Initialize the training loop
    while current_frame_step <= max_frame_steps:
        current_state = env.reset()
        current_state = current_state[0]
        done = False
        total_reward = 0
        for i in range(max_t):
            # Select an action
            if current_frame_step < start_steps:
                action = env.action_space.sample()
            else:
                current_state_tensor = torch.tensor([current_state], device=device, dtype=torch.float32)
                action_tensor, _, _, _ = actor.sample_action(current_state_tensor, reparameterize=False)
                action = action_tensor.cpu().detach().numpy().copy().squeeze()
            # Perform the action
            next_state, reward, done, _, info = env.step(action)
            
            # Add experience transition to replay buffer, but if we are using curriculum learning, 
            # then calculate temporal difference error and save it as well
            if use_curriculum:
                # Calculate the temporal difference error
                with torch.no_grad():
                    next_state_tensor = torch.tensor([next_state], device=device, dtype=torch.float32)
                    actions, log_probs, _, _ = actor.sample_action(next_state_tensor, reparameterize=True)
                    log_probs = log_probs.view(-1)
                    q1_next_target = critic_1_target(next_state_tensor, actions)
                    q2_next_target = critic_2_target(next_state_tensor, actions)
                    min_next_value = torch.min(q1_next_target, q2_next_target) # Proposed in the TD3 paper to deal with overestimation bias
                    min_next_value = min_next_value.view(-1)
                    min_next_value -= alpha_factor.alpha * log_probs
                    current_q_1 = critic_1(torch.tensor([current_state_tensor], device=device, dtype=torch.float32), action_tensor)
                    current_q_2 = critic_2(torch.tensor([current_state_tensor], device=device, dtype=torch.float32), action_tensor)
                    current_q = torch.min(current_q_1, current_q_2) # To minimize overestimation
                    td_error = ((reward + gamma * min_next_value - current_q)**2).item()
                # Add the experience transition to the replay buffer
                replay_buffer.add((current_state, action, reward, next_state, done, td_error))
            else:
                replay_buffer.add((current_state, action, reward, next_state, done))
                
            # Update the current state
            current_state = next_state.copy()

            ###### Logging to wandb
            # Evaluate the agent every 3000 steps
            if current_frame_step % 3000 == 0:
                env_eval.reset(seed=42)
                rewards_eval = []
                for _ in range(3):
                    current_state_eval = env_eval.reset()[0]
                    done_eval = False
                    reward_acc = 0
                    for k in range(max_t):
                        action_eval, _, mean, std = actor.sample_action(torch.tensor([current_state_eval], device=device, dtype=torch.float32), reparameterize=False)
                        action_eval_numpy_copy = action_eval.cpu().detach().numpy().copy().squeeze()
                        next_state_eval, reward_eval, done_eval, _, info = env_eval.step(action_eval_numpy_copy)
                        current_state_eval = next_state_eval
                        reward_acc += reward_eval
                        if done_eval:
                            break
                    rewards_eval.append(reward_acc)
                mean_reward_eval = sum(rewards_eval)/len(rewards_eval)
                std_reward_eval = np.std(rewards_eval).item()
                mean_reward_eval_smoothed.append(mean_reward_eval)
                average_reward_eval = sum(mean_reward_eval_smoothed)/len(mean_reward_eval_smoothed)
                # Calculating the gradients of the networks
                squared_norm_gradients_actor = 0
                for w in actor.parameters():
                    if w.grad is None:
                        break
                    else:
                        squared_norm_gradients_actor += (torch.norm(w.grad)**2).item()
                squared_norm_gradients_critic_1 = 0
                for w in critic_1.parameters():
                    if w.grad is None:
                        break
                    else:
                        squared_norm_gradients_critic_1 += (torch.norm(w.grad)**2).item()
                squared_norm_gradients_critic_2 = 0
                for w in critic_2.parameters():
                    if w.grad is None:
                        break
                    else:
                        squared_norm_gradients_critic_2 += (torch.norm(w.grad)**2).item()

                squared_norm_gradients_actor = np.sqrt(squared_norm_gradients_actor)
                squared_norm_gradients_critic_1 = np.sqrt(squared_norm_gradients_critic_1)
                squared_norm_gradients_critic_2 = np.sqrt(squared_norm_gradients_critic_2)
                # Logging metrics to wandb
                log_dict = {
                    "train/step": current_frame_step,
                    "train/training_reward": reward,
                    "train/training_episode": current_episode,
                    "train/squared_norm_gradients_actor": squared_norm_gradients_actor,
                    "train/squared_norm_gradients_critic_1": squared_norm_gradients_critic_1,
                    "train/squared_norm_gradients_critic_2": squared_norm_gradients_critic_2,
                    "train/action_taken_1": action[0],
                    "train/action_taken_2": action[1],
                    "train/action_taken_3": action[2],
                    "train/action_taken_4": action[3],
                    "train/actor_loss": actor_loss,
                    "train/critic_loss_1": critic_loss_1,
                    "train/critic_loss_2": critic_loss_2,
                    "train/alpha_loss": alpha_loss,
                    "train/alpha_value": alpha_value,
                    "train/q1_mean": q1_mean,
                    "train/q2_mean": q2_mean,
                    "train/replay_buffer_#ofexperiences": len(replay_buffer.buffer_deque),
                    "train/mean_reward_eval": mean_reward_eval, 
                    "train/mean_reward_eval_smoothed": sum(mean_reward_eval_smoothed)/len(mean_reward_eval_smoothed), 
                    "train/std_reward_eval": std_reward_eval
                }
                if use_curriculum:
                    log_dict["train/curriculum"] = replay_buffer.current_curriculum
                run.log(log_dict)
                ###### End of logging to wandb of inner loop
            
            total_reward += reward #  Accumulate reward of a training episode
            current_frame_step += 1 #  Train step counter
            try:
                next(progress_bar)
            except:
                save_model_and_log_final_metrics()
                return
            if average_reward_eval > 300: #  It is considered to be solved if the average of the last 100 evaluated episodes is over 300
                save_model_and_log_final_metrics()
                return
            # Check if the episode is done
            if done:
                break
        # Perform optimization step after the episode is done
        if len(replay_buffer.buffer_deque) > batch_size and current_frame_step > start_steps:
            for _ in range(functions_updates):
                # Get the minibatch from the replay buffer
                minibatch = replay_buffer.sample(batch_size) 
                actor_loss, critic_loss_1, critic_loss_2, alpha_loss, q1_mean, q2_mean, alpha_value = perform_optimization_step(actor, critic_1, critic_2, critic_1_target, critic_2_target, 
                                                                                                                                alpha_factor, minibatch, gamma, tau, device, grad_clip_value, use_curriculum)
                current_optimization_steps += 1
        current_episode += 1 
        ## Outer loop logging
        # Logging the total reward of the training episode
        log_dict = {
            "train_ep/episode_reward": total_reward,
            "train_ep/episode": current_episode,
            "train_ep/mean_reward_eval": mean_reward_eval,
            "train_ep/mean_reward_eval_smoothed": average_reward_eval,
            "train_ep/std_reward_eval": std_reward_eval,
            "train_ep/current_optimization_steps": current_optimization_steps,
        }
        run.log(log_dict)
    
    # Post-training code
    run.save
    run.finish()
    if save_agent:
        torch.save(actor.state_dict(), 'experimenting/models/trained_models/bipedal_walker/' + run_name + '.pt')

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == "__main__":
    from distutils.util import strtobool

    parser = argparse.ArgumentParser(description='Arguments for training a SAC agent on the BipedalWalker environment')
    # Misc arguments
    parser.add_argument('--name_env', type=str, default="BipedalWalker-v3", help='Name of the environment')
    parser.add_argument('-gpu', '--gpu', metavar='GPU', type=str, help='gpu to use', default='1') # Only 3 and 4 should be used. Number 2 could also be used but check availability first
    # Rest of the arguments
    parser.add_argument('--alpha', type=float, default=0.0003, help='Learning rate for the actor')
    parser.add_argument('--beta', type=float, default=0.0003, help='Learning rate for the critic')
    parser.add_argument('--theta', type=float, default=0.0003, help='Learning rate for the q network')
    parser.add_argument('--use_curriculum', type=lambda x: bool(strtobool(x)), default=False, help='Whether to use curriculum learning')
    parser.add_argument('--max_frame_steps', type=int, default=4000000, help='Maximum number of frame iterations where one optimization step is performed')
    parser.add_argument('--max_t', type=int, default=1600, help='Maximum number of steps per episode')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for the replay buffer')
    parser.add_argument('--start_steps', type=int, default=10000, help='Number of steps to take before starting to optimize the networks')
    parser.add_argument('--functions_updates', type=int, default=2, help='Number of optimization steps to take per update frequency')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.01, help='Soft update parameter for the target networks')
    parser.add_argument('--grad_clip_value', type=float, default=10.0, help='Value to clip the gradients to')
    parser.add_argument('--n_neurons_first_layer', type=int, default=256, help='Number of neurons in the first layer')
    parser.add_argument('--n_neurons_second_layer', type=int, default=256, help='Number of neurons in the second layer')
    parser.add_argument('--log_std_max', type=float, default=2.0, help='Maximum value for the log standard deviation of the Gaussian distribution')
    parser.add_argument('--buffer_size', type=int, default=1000000, help='Size of the replay buffer')
    parser.add_argument('--project_name', type=str, default="sac_bipedal_walker", help='Name of the wandb project')
    parser.add_argument('--number_of_curriculums', type=int, default=5, help='Number of curriculums to use')
    parser.add_argument('--anti_curriculum', type=lambda x: bool(strtobool(x)), default=False, help='Whether to use anti-curriculum')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--save_agent', type=lambda x: bool(strtobool(x)), default=False, help='Whether to save the agent or not')
    args = parser.parse_args()

    name_env = args.name_env
    gpu = args.gpu
    alpha = args.alpha
    beta = args.beta
    theta = args.theta
    use_curriculum = args.use_curriculum
    max_frame_steps = args.max_frame_steps
    max_t = args.max_t
    batch_size = args.batch_size
    start_steps = args.start_steps
    functions_updates = args.functions_updates
    gamma = args.gamma
    tau = args.tau
    grad_clip_value = args.grad_clip_value
    n_neurons_first_layer = args.n_neurons_first_layer
    n_neurons_second_layer = args.n_neurons_second_layer
    log_std_max = args.log_std_max
    buffer_size = args.buffer_size
    project_name = args.project_name
    number_of_curriculums = args.number_of_curriculums
    anti_curriculum = args.anti_curriculum
    seed = args.seed
    save_agent = args.save_agent

    set_seed(seed)

    train_sac_bipedal_walker(name_env, gpu, alpha, beta, theta, use_curriculum, max_frame_steps, max_t, batch_size, 
                             start_steps, functions_updates, gamma, tau, grad_clip_value, n_neurons_first_layer, 
                             n_neurons_second_layer, log_std_max, buffer_size, project_name, 
                             number_of_curriculums, anti_curriculum, seed, save_agent)