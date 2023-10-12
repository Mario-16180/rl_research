from collections import deque
from rl_utils.stack_frames import stacked_frames_class
import numpy as np
import torch
import pandas as pd

class memory_with_curriculum():
    def __init__(self, max_size):
        self.buffer_size = max_size
        self.buffer_deque = deque(maxlen = max_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def add(self, experience):
        self.buffer_deque.append(experience)
    
    def sample(self, batch_size):
        index = np.random.choice(np.arange(len(self.buffer_deque)), size = batch_size, replace = False)
        return [self.buffer_deque[i] for i in index]

    def populate_memory_model(self, model, environment, name_env, k_initial_experiences):
        rewardbounds_per_env=pd.read_csv('rl_utils/reward_data_per_environment.csv', delimiter=' ', header=0)
        min_r = rewardbounds_per_env[rewardbounds_per_env.Environment == name_env].Rminhard.item()
        max_r = rewardbounds_per_env[rewardbounds_per_env.Environment == name_env].Rmaxhard.item()
        normalize_reward = lambda r: (r - min_r) / (max_r - max_r)
        for i in range(k_initial_experiences):
            # First initializing transition
            if i == 0:
                stacked_frames_object = stacked_frames_class()
                initial_frame = environment.reset()
                stacked_frames_object.initialize_stack(initial_frame)
                
            # Getting a random action from action space
            action, _ = model.select_action(environment, stacked_frames_object.stacked_frames_array, 0.05, 1, 0.05, 0, self.device)
            action = action.item()
            next_frame, reward, done, _ = environment.step(action)
            reward = normalize_reward(reward)
            # Save current state
            current_stacked_frames = stacked_frames_object.stacked_frames_array
        
            # If it's the end of the episode
            if done:
                # End of episode
                next_frame = np.zeros((64,64,3), dtype=np.uint8)
                stacked_frames_object.append_frame_to_stack(next_frame)
                next_stacked_frames = stacked_frames_object.stacked_frames_array
                
                # Add experience to replay memory
                self.add((current_stacked_frames, action, reward, next_stacked_frames, done))
                
                # Start a new episode
                initial_frame = environment.reset()
                
                # Initialize stack
                stacked_frames_object.initialize_stack(initial_frame)
                
            else:
                stacked_frames_object.append_frame_to_stack(next_frame)
                next_stacked_frames = stacked_frames_object.stacked_frames_array

                # Add experience to replay memory
                self.add((current_stacked_frames, action, reward, next_stacked_frames, done))

class memory():
    def __init__(self, max_size):
        self.buffer_size = max_size
        self.buffer_deque = deque(maxlen = max_size)
    
    def add(self, experience):
        self.buffer_deque.append(experience)
    
    def sample(self, batch_size):
        index = np.random.choice(np.arange(len(self.buffer_deque)), size = batch_size, replace = False)
        return [self.buffer_deque[i] for i in index]

    def populate_memory_random(self, environment, name_env, k_initial_experiences):
        rewardbounds_per_env=pd.read_csv('rl_utils/reward_data_per_environment.csv', delimiter=' ', header=0)
        min_r = rewardbounds_per_env[rewardbounds_per_env.Environment == name_env].Rminhard.item()
        max_r = rewardbounds_per_env[rewardbounds_per_env.Environment == name_env].Rmaxhard.item()
        normalize_reward = lambda r: (r - min_r) / (max_r - max_r)
        for i in range(k_initial_experiences):
            # First initializing transition
            if i == 0:
                stacked_frames_object = stacked_frames_class()
                initial_frame = environment.reset()
                stacked_frames_object.initialize_stack(initial_frame)
                
            # Getting a random action from action space
            action = environment.action_space.sample()
            next_frame, reward, done, _ = environment.step(action)
            reward = normalize_reward(reward)
            # Save current state
            current_stacked_frames = stacked_frames_object.stacked_frames_array
        
            # If it's the end of the episode
            if done:
                # End of episode
                next_frame = np.zeros((64,64,3), dtype=np.uint8)
                stacked_frames_object.append_frame_to_stack(next_frame)
                next_stacked_frames = stacked_frames_object.stacked_frames_array
                
                # Add experience to replay memory
                self.add((current_stacked_frames, action, reward, next_stacked_frames, done))
                
                # Start a new episode
                initial_frame = environment.reset()
                
                # Initialize stack
                stacked_frames_object.initialize_stack(initial_frame)
                
            else:
                stacked_frames_object.append_frame_to_stack(next_frame)
                next_stacked_frames = stacked_frames_object.stacked_frames_array

                # Add experience to replay memory
                self.add((current_stacked_frames, action, reward, next_stacked_frames, done))