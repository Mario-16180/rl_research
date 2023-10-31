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

def train_dqn_curriculum(name_env, episodes, batch_size, gamma, epsilon_start, epsilon_decay, epsilon_min, learning_rate, tau, 
                      num_levels, num_levels_eval, start_level, start_level_test, background,
                      initial_random_experiences, memory_capacity, resume, project_name,
                      number_of_curriculums, curriculum, difficulty, gpu):
    # Used to normalize the reward
    rewardbounds_per_env=pd.read_csv('experimenting/rl_utils/reward_data_per_environment.csv', delimiter=' ', header=0)
    if difficulty == 'easy':
        min_r = rewardbounds_per_env[rewardbounds_per_env.Environment == name_env].Rmineasy.item()
        max_r = rewardbounds_per_env[rewardbounds_per_env.Environment == name_env].Rmaxeasy.item()
    elif difficulty == 'hard':
        min_r = rewardbounds_per_env[rewardbounds_per_env.Environment == name_env].Rminhard.item()
        max_r = rewardbounds_per_env[rewardbounds_per_env.Environment == name_env].Rmaxhard.item()
    normalize_reward = lambda r: (r - min_r) / (max_r - min_r)

    # Initialize environment for training and evaluation
    env = gym.make(name_env, start_level=start_level, num_levels=num_levels, use_backgrounds=background, distribution_mode=difficulty)
    env_eval = gym.make(name_env, start_level=start_level_test, num_levels=num_levels_eval, use_backgrounds=background, distribution_mode=difficulty)

    # Choose GPU if available
    device = torch.device("cuda:"+gpu if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model and target model
    model_policy = impala_cnn(env).to(device)
    model_target = impala_cnn(env).to(device)
    model_target.load_state_dict(model_policy.state_dict())

    # Initialize replay buffer if it's empty
    if curriculum:
        replay_buffer = memory_with_curriculum(max_size=memory_capacity, curriculums=number_of_curriculums)
        replay_buffer.populate_memory_model(model_policy, env, name_env, k_initial_experiences=initial_random_experiences, device=device)
    else:
        replay_buffer = memory(max_size=memory_capacity)
        replay_buffer.populate_memory_random(env, name_env, k_initial_experiences=initial_random_experiences)
    optimizer = torch.optim.Adam(model_policy.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
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
    "curriculum": curriculum,
    "number_of_curriculums": number_of_curriculums,
    "difficulty": difficulty,},
    resume=resume,
    )
    run.define_metric("train/step")
    run.define_metric("train/*", step_metric="train/step")
    run_name = run.name + name_env
    current_episode = 0
    current_step = 0
    if wandb.run.resumed:
        checkpoint = torch.load(wandb.restore('models/trained_models/checkpoint.tar'))
        model_policy.load_state_dict(checkpoint['model_state_dict'])
        model_target.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        current_episode = checkpoint['episode']
        current_step = checkpoint['step']
        loss = checkpoint['loss']
        replay_buffer = pickle.load(open('models/trained_models/checkpoint_buffer', 'rb'))
    ##### End of wandb login
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
    mean_reward_eval_smoothed = deque(maxlen=50)
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

            loss = perform_optimization_step(model_policy, model_target, minibatch, gamma, optimizer, criterion, device, batch_size, curriculum=curriculum)

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
                rewards = []
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
                    rewards.append(reward_acc)
                mean_reward_eval = sum(rewards)/len(rewards)
                mean_reward_eval_smoothed.append(mean_reward_eval)
                run.log({"train/mean_reward_eval": mean_reward_eval, "train/mean_reward_eval_smoothed": sum(mean_reward_eval_smoothed)/len(mean_reward_eval_smoothed)})
                model_policy.save_model(episode=episode, train_step=current_step, optimizer=optimizer, loss=loss, buffer=replay_buffer, path='experimenting/models/trained_models/checkpoint')

            # Logging metrics to wandb
            squared_norm_gradients = 0
            for w in model_policy.parameters():
                squared_norm_gradients += (torch.norm(w.grad)**2).item()
            log_dict = {
                "train/step": current_step,
                "train/training_reward": reward,
                "train/training_episode": episode,
                "train/loss": loss,
                "train/squared_norm_gradients": squared_norm_gradients,
                "train/action_taken": action,
                "train/epsilon": eps,
                "train/replay_buffer_#ofexperiences": len(replay_buffer.buffer_deque),
                #"train/learning_rate": scheduler.get_last_lr()[0],
            }
            if curriculum:
                log_dict["train/curriculum"] = replay_buffer.curriculum
            run.log(log_dict)
        run.log({"train/episode_reward": train_reward})
        #scheduler.step()
    run.save
    run.finish()
    torch.save(model_policy.state_dict(), 'experimenting/models/trained_models/' + run_name + '.pt')

if __name__ == '__main__':
    environment_names_dictionary = {
        "coinrun": "procgen:procgen-coinrun-v0",
        "starpilot": "procgen:procgen-starpilot-v0",
        "caveflyer": "procgen:procgen-caveflyer-v0",
        "dodgeball": "procgen:procgen-dodgeball-v0",
        "fruitbot": "procgen:procgen-fruitbot-v0",
        "chaser": "procgen:procgen-chaser-v0",
        "miner": "procgen:procgen-miner-v0",
        "jumper": "procgen:procgen-jumper-v0",
        "leaper": "procgen:procgen-leaper-v0",
        "maze": "procgen:procgen-maze-v0",
        "bigfish": "procgen:procgen-bigfish-v0",
        "heist": "procgen:procgen-heist-v0",
        "climber": "procgen:procgen-climber-v0",
        "plunder": "procgen:procgen-plunder-v0",
        "ninja": "procgen:procgen-ninja-v0",
        "bossfight": "procgen:procgen-bossfight-v0",
    }
    parser = argparse.ArgumentParser(description='Arguments for training the DQN agent')
    parser.add_argument('-env','--name_env', metavar='N', type=str, help='name of the environment', default='bossfight')
    parser.add_argument('-e', '--episodes', metavar='E', type=int, help='number of training episodes', default=5000)
    parser.add_argument('-bs', '--batch_size', metavar='B', type=int, help='number of transitions to train from the replay buffer',default=64)
    parser.add_argument('-g', '--gamma', metavar='G', type=float, help='discount factor', default=0.99)
    parser.add_argument('-epss', '--epsilon_start', metavar='ES', type=float, help='initial value of epsilon', default=1.0)
    parser.add_argument('-epsd', '--epsilon_decay', metavar='ED', type=float, help='decay rate of epsilon', default=100000)
    parser.add_argument('-epsm', '--epsilon_min', metavar='EM', type=float, help='minimum value of epsilon', default=0.1)
    parser.add_argument('-lr', '--learning_rate', metavar='LR', type=float, help='learning rate', default=0.0001)
    parser.add_argument('-t', '--tau', metavar='T', type=float, help='parameter for updating the target network', default=0.001)
    parser.add_argument('-nl', '--num_levels', metavar='NL', type=int, help='number of levels in the environment', default=200)
    parser.add_argument('-nle', '--num_levels_eval', metavar='NLE', type=int, help='number of levels in the evaluation environment', default=20)
    parser.add_argument('-sl', '--start_level', metavar='SL', type=int, help='starting level for training', default=0)
    parser.add_argument('-slt', '--start_level_test', metavar='SLT', type=int, help='starting level for evaluation', default=516)
    parser.add_argument('-b', '--background', metavar='BG', type=bool, help='use background in the environment', default=True)
    parser.add_argument('-ire', '--initial_random_experiences', metavar='IRE', type=int, help='number of initial random experiences', default=5000)
    parser.add_argument('-mc', '--memory_capacity', metavar='MC', type=int, help='size of the replay buffer', default=50000)
    parser.add_argument('-r', '--resume', metavar='R', type=bool, help='resume training', default=False)
    parser.add_argument('-pn', '--project_name', metavar='PN', type=str, help='name of the project in wandb', default='rl_research_mbzuai')
    parser.add_argument('-cn', '--number_of_curriculums', metavar='NC', type=int, help='number of curriculums', default=3)
    parser.add_argument('-c', '--curriculum', metavar='C', type=bool, help='use curriculum learning', default=False)
    parser.add_argument('-d', '--difficulty', metavar='D', type=str, help='difficulty of the environment', default='easy')
    parser.add_argument('-gpu', '--gpu', metavar='GPU', type=str, help='gpu to use', default='3') # Only 3 and 4 should be used. Number 2 could also be used but check availability first
    args = parser.parse_args()

    name_env = environment_names_dictionary[args.name_env]
    episodes = args.episodes
    batch_size = args.batch_size
    gamma = args.gamma
    epsilon_decay = args.epsilon_decay
    epsilon_start = args.epsilon_start
    epsilon_min = args.epsilon_min
    learning_rate = args.learning_rate
    tau = args.tau
    num_levels = args.num_levels
    num_levels_eval = args.num_levels_eval
    start_level = args.start_level
    start_level_test = args.start_level_test
    background = args.background
    initial_random_experiences = args.initial_random_experiences
    memory_capacity = args.memory_capacity
    resume = args.resume
    project_name = args.project_name
    number_of_curriculums = args.number_of_curriculums
    curriculum = args.curriculum
    difficulty = args.difficulty
    gpu = args.gpu

    train_dqn_curriculum(name_env=name_env, episodes=episodes, batch_size=batch_size, gamma=gamma, epsilon_start=epsilon_start, 
                        epsilon_decay=epsilon_decay, epsilon_min=epsilon_min,
                        learning_rate=learning_rate, tau=tau, num_levels=num_levels, 
                        num_levels_eval=num_levels_eval, start_level=start_level, 
                        start_level_test=start_level_test, background=background, initial_random_experiences=initial_random_experiences, 
                        memory_capacity=memory_capacity, resume=resume, project_name=project_name,
                        number_of_curriculums=number_of_curriculums, curriculum=curriculum, difficulty=difficulty, gpu=gpu)

    # According to wandb, I'm using a maximum of 8 gb of GPU memory per job run. For the last sweep, somebody was usisng the same GPU as I was, it saturated
    # and therefore I got an error and the process was terminated.