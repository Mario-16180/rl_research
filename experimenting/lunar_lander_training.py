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
from models.lunar_lander_dqn_architecture import lunar_lander_mlp
from rl_utils.replay_buffer import memory_lunar_lander as memory
from rl_utils.optimization_dqn import perform_optimization_step
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def train_dqn_lunar_lander(name_env, max_optimization_steps, max_steps, batch_size, grad_clip_value, gamma, epsilon_start, epsilon_decay, epsilon_min, learning_rate, tau,
                        first_layer_neurons, second_layer_neurons, initial_random_experiences, memory_capacity, project_name, max_train_steps_per_curriculum, 
                        number_of_curriculums, use_curriculum, gpu, anti_curriculum, percentile, save_agent, seed, criterion_name):
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

    # Initialize model and target model
    model_policy = lunar_lander_mlp(env, first_layer_neurons, second_layer_neurons).to(device)
    model_target = lunar_lander_mlp(env, first_layer_neurons, second_layer_neurons).to(device)
    model_target.load_state_dict(model_policy.state_dict())

    # Initialize replay buffer if it's empty
    replay_buffer = memory(buffer_size=memory_capacity, max_train_steps_per_curriculum=max_train_steps_per_curriculum, 
                           use_curriculum=use_curriculum, n_curriculums=number_of_curriculums, anti_curriculum=anti_curriculum, percentile=percentile)

    replay_buffer.populate_memory_random(env, k_initial_experiences=initial_random_experiences, gamma=gamma, model=model_policy, device=device)

    optimizer = torch.optim.Adam(model_policy.parameters(), lr=learning_rate)

    if criterion_name == 'MSE':
        criterion = nn.MSELoss()
    elif criterion_name == 'Huber':
        criterion = nn.HuberLoss()

    # Auxiliar function to save the model and log final metrics
    def save_model_and_log_final_metrics():
        # Post-training code
        # Logging the total reward of the training episode
        log_dict = {
            "train_ep/episode_reward": train_reward,
            "train_ep/episode": current_episode,
            "train_ep/mean_reward_eval": mean_reward_eval,
            "train_ep/mean_reward_eval_smoothed": average_reward_eval,
            "train_ep/std_reward_eval": std_reward_eval
        }
        run.log(log_dict)
        run.save
        run.finish()
        if save_agent:
            torch.save(model_policy.state_dict(), 'experimenting/models/trained_models/lunar_lander/' + run_name + '.pt')

    #### Login wandb + Hyperparameters and metadata
    # Start a new wandb run to track this script
    run = wandb.init(
    # Set the wandb project where this run will be logged
    project=project_name,
    # Track hyperparameters and run metadata
    config={
    "architecture": "Lunar lander basic MLP",
    "environment": name_env,
    "learning_rate": learning_rate,
    "max_optimization_steps": max_optimization_steps,
    "batch_size": batch_size,
    "gamma": gamma,
    "grad_clip_value": grad_clip_value,
    "first_layer_neurons": first_layer_neurons,
    "second_layer_neurons": second_layer_neurons,
    "initial_random_experiences": initial_random_experiences,
    "epsilon_start": epsilon_start,
    "epsilon_decay": epsilon_decay,
    "epsilon_min": epsilon_min,
    "tau": tau,
    "memory_length": replay_buffer.buffer_size,
    "optimizer": "Adam",
    "criterion": criterion_name,
    "device": device,
    "use_curriculum": replay_buffer.use_curriculum,
    "n_curriculums": replay_buffer.n_curriculums,
    "anti_curriculum": replay_buffer.anti_curriculum,
    "seed": seed})
    run.define_metric("train/step")
    run.define_metric("train_ep/episode")
    run.define_metric("train/*", step_metric="train/step")
    run.define_metric("train_ep/*", step_metric="train_ep/episode")
    run_name = run.name + name_env

    current_episode = 0
    current_step = 0
    ##### End of wandb login
    mean_reward_eval_smoothed = deque(maxlen=100)
    mean_reward_eval = 0
    average_reward_eval = 0
    std_reward_eval = 0
    progress_bar = iter(tqdm(range(max_optimization_steps)))
    # Training loop
    while current_step <= max_optimization_steps:
        current_state = env.reset()[0]
        done = False
        train_reward = 0
        for i in range(max_steps):
            action, eps = model_policy.select_action(env, current_state, epsilon_start, epsilon_decay, epsilon_min, current_step, device)
            next_state, reward, done, _, info = env.step(action)
            minibatch = replay_buffer.sample(batch_size)
            loss = perform_optimization_step(model_policy, model_target, minibatch, gamma, optimizer, criterion, device, batch_size, grad_clip_value=grad_clip_value, curriculum=curriculum)
            if curriculum: 
                # Calculate temporal difference
                with torch.no_grad():
                    next_state = torch.tensor([next_state], device=device, dtype=torch.float32)
                    current_state = torch.tensor([current_state], device=device, dtype=torch.float32).reshape(1,8)
                    temporal_difference = ((reward + gamma * model_policy.forward(next_state).max(1)[0].item() - 
                                            model_policy.forward(current_state)[0,action])**2).item()
                # Convert tensors to numpy arrays before adding them to the replay buffer
                next_state = next_state.cpu().numpy()
                current_state = current_state.cpu().numpy()
                replay_buffer.add((current_state, action, reward, next_state, done, temporal_difference))
            else:
                replay_buffer.add((current_state, action, reward, next_state, done))
            
            # Update target network
            policy_weights = model_policy.state_dict()
            target_weights = model_target.state_dict()
            for name in policy_weights:
                target_weights[name] = tau * policy_weights[name] + (1 - tau) * target_weights[name]
            model_target.load_state_dict(target_weights)
            
            # Update current state
            current_state = next_state.copy()

            # Evaluate the agent every 500 steps
            if current_step % 500 == 0:
                env_eval.reset(seed=42)
                rewards_eval = []
                for _ in range(5):
                    current_state_eval = env_eval.reset()[0]
                    done_eval = False
                    reward_acc = 0
                    for k in range(max_steps):
                        action_eval, _ = model_policy.select_action(env_eval, current_state_eval, 0, 1, 0, current_step, device)
                        next_state_eval, reward_eval, done_eval, _, info = env_eval.step(action_eval)
                        current_state_eval = next_state_eval
                        reward_acc += reward_eval
                        if done_eval:
                            break
                    rewards_eval.append(reward_acc)
                mean_reward_eval = sum(rewards_eval)/len(rewards_eval)
                std_reward_eval = np.std(rewards_eval).item()
                mean_reward_eval_smoothed.append(mean_reward_eval)
                average_reward_eval = sum(mean_reward_eval_smoothed)/len(mean_reward_eval_smoothed)
                # Logging metrics to wandb
                squared_norm_gradients = 0
                for w in model_policy.parameters():
                    squared_norm_gradients += (torch.norm(w.grad)**2).item()
                log_dict = {
                    "train/step": current_step,
                    "train/training_reward": reward,
                    "train/training_episode": current_episode,
                    "train/loss": loss,
                    "train/squared_norm_gradients": squared_norm_gradients,
                    "train/action_taken": action,
                    "train/epsilon": eps,
                    "train/replay_buffer_#ofexperiences": len(replay_buffer.buffer_deque),
                    "train/mean_reward_eval": mean_reward_eval, 
                    "train/mean_reward_eval_smoothed": average_reward_eval, 
                    "train/std_reward_eval": std_reward_eval
                }
                if curriculum:
                    log_dict["train/curriculum"] = replay_buffer.curriculum
                run.log(log_dict)

            # Counters
            current_step += 1 # Training step
            train_reward += reward
            try:
                next(progress_bar)
            except:
                save_model_and_log_final_metrics()
                return
            if average_reward_eval > 200:
                save_model_and_log_final_metrics()
                return
            # If the episode is done, break the loop
            if done:
                break
        current_episode += 1
        # Logging the total reward of the training episode
        log_dict = {
            "train_ep/episode_reward": train_reward,
            "train_ep/episode": current_episode,
            "train_ep/mean_reward_eval": mean_reward_eval,
            "train_ep/mean_reward_eval_smoothed": average_reward_eval,
            "train_ep/std_reward_eval": std_reward_eval
        }
        run.log(log_dict)
    # Post-training code
    run.save
    run.finish()
    if save_agent:
        torch.save(model_policy.state_dict(), 'experimenting/models/trained_models/lunar_lander/' + run_name + '.pt')

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# This script is used to do a fast training in the LunarLander-v2 environment
if __name__ == '__main__':
    from distutils.util import strtobool

    parser = argparse.ArgumentParser(description='Arguments for training the DQN agent')
    # Training related arguments
    parser.add_argument('-mos', '--max_optimization_steps', metavar='E', type=int, help='maximum number of training steps, i.e. when the function optimization is called', default=300000)
    parser.add_argument('-ms', '--max_steps', metavar='MS', type=int, help='maximum number of steps per episode', default=1000)
    parser.add_argument('-bs', '--batch_size', metavar='B', type=int, help='number of transitions to train from the replay buffer',default=64)
    parser.add_argument('-gc', '--grad_clip_value', metavar='GC', type=float, help='value to clip the gradients', default=1.5) 
    # Reinforcement learning related arguments
    parser.add_argument('-g', '--gamma', metavar='G', type=float, help='discount factor', default=0.9995)
    parser.add_argument('-epss', '--epsilon_start', metavar='ES', type=float, help='initial value of epsilon', default=1.0)
    parser.add_argument('-epsm', '--epsilon_min', metavar='EM', type=float, help='minimum value of epsilon', default=0)
    parser.add_argument('-epsd', '--epsilon_decay', metavar='ED', type=float, help='decay rate of epsilon', default=0.000015)
    parser.add_argument('-lr', '--learning_rate', metavar='LR', type=float, help='learning rate', default=0.0015)
    parser.add_argument('-t', '--tau', metavar='T', type=float, help='parameter for updating the target network', default=0.01)
    parser.add_argument('-cn', '--criterion_name', metavar='CN', type=str, help='criterion to use for the loss', default='MSE')
    # Hyperparameters for the neural network
    parser.add_argument('-fln', '--first_layer_neurons', metavar='FLN', type=int, help='number of neurons in the first layer', default=256)
    parser.add_argument('-sln', '--second_layer_neurons', metavar='SLN', type=int, help='number of neurons in the second layer', default=256)
    # Replay buffer related arguments
    parser.add_argument('-ire', '--initial_random_experiences', metavar='IRE', type=int, help='number of initial random experiences', default=8192)
    parser.add_argument('-mc', '--memory_capacity', metavar='MC', type=int, help='size of the replay buffer', default=100000)
    # Curriculum related arguments
    parser.add_argument('-nc', '--number_of_curriculums', metavar='NC', type=int, help='number of curriculums', default=5)
    parser.add_argument('-c', '--curriculum', metavar='C', type=lambda x: bool(strtobool(x)), help='use curriculum learning', default=False)
    parser.add_argument('-ac', '--anti_curriculum', metavar='AC', type=lambda x: bool(strtobool(x)), help='going from easy to hard = false, hard to easy = true', default=False)
    parser.add_argument('-mcs', '--max_train_steps_per_curriculum', metavar='MCS', type=int, help='maximum number of training steps per curriculum', default=1000)
    parser.add_argument('-p', '--percentile', metavar='P', type=int, help='percentage of already sorted top experiences, if 0, then all experiences will be considered, if 50, only the second half of orderd experiences is considered', default=0)
    # Misc arguments
    parser.add_argument('-env','--name_env', metavar='N', type=str, help='name of the environment', default='LunarLander-v2')
    parser.add_argument('-pn', '--project_name', metavar='PN', type=str, help='name of the project in wandb', default='lunar_lander_final')
    parser.add_argument('-gpu', '--gpu', metavar='GPU', type=str, help='gpu to use', default='3') # Only 3 and 4 should be used. Number 2 could also be used but check availability first
    parser.add_argument('-sa', '--save_agent', metavar='SA', type=lambda x: bool(strtobool(x)), help='save the agent', default=False)
    parser.add_argument('-seed', '--seed', metavar='SEED', type=int, help='seed for reproducibility', default=42)
    args = parser.parse_args()

    # Training related arguments
    max_optimization_steps = args.max_optimization_steps
    max_steps = args.max_steps
    batch_size = args.batch_size
    grad_clip_value = args.grad_clip_value
    # Reinforcement learning related arguments
    gamma = args.gamma
    epsilon_decay = args.epsilon_decay
    epsilon_start = args.epsilon_start
    epsilon_min = args.epsilon_min
    learning_rate = args.learning_rate
    tau = args.tau
    criterion_name = args.criterion_name
    # Neural network architecture related arguments
    first_layer_neurons = args.first_layer_neurons
    second_layer_neurons = args.second_layer_neurons
    # Replay buffer related arguments
    initial_random_experiences = args.initial_random_experiences
    memory_capacity = args.memory_capacity
    # Curriculum related arguments
    number_of_curriculums = args.number_of_curriculums
    curriculum = args.curriculum
    anti_curriculum = args.anti_curriculum
    max_train_steps_per_curriculum = args.max_train_steps_per_curriculum
    percentile = args.percentile
    # Misc arguments
    name_env = args.name_env
    project_name = args.project_name
    gpu = args.gpu
    save_agent = args.save_agent
    seed = args.seed
    
    set_seed(seed)

    train_dqn_lunar_lander(name_env=name_env, max_optimization_steps=max_optimization_steps, max_steps=max_steps, batch_size=batch_size, grad_clip_value=grad_clip_value,
                           gamma=gamma, epsilon_start=epsilon_start, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min,
                            learning_rate=learning_rate, tau=tau, first_layer_neurons=first_layer_neurons, second_layer_neurons=second_layer_neurons, 
                            initial_random_experiences=initial_random_experiences, 
                            memory_capacity=memory_capacity, max_train_steps_per_curriculum=max_train_steps_per_curriculum,
                            percentile=percentile, project_name=project_name,
                            number_of_curriculums=number_of_curriculums, use_curriculum=curriculum, anti_curriculum=anti_curriculum, 
                            gpu=gpu, save_agent=save_agent, seed=seed, criterion_name=criterion_name)