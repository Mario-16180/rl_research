from collections import deque
from rl_utils.stack_frames import stacked_frames_class
import numpy as np
import torch
import pandas as pd

class memory_with_curriculum():
    def __init__(self, max_size, gamma=0.99, curriculums=3):
        self.buffer_size = max_size
        self.buffer_deque = deque(maxlen = max_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.number_of_curriculums = curriculums
        self.flag_first_curriculum = True
        self.counter = 0
    
    def add(self, experience):
        self.buffer_deque.append(experience)
    
    def sample(self, batch_size):
        if self.flag_first_curriculum:
            self.make_curriculums()
            self.flag_first_curriculum = False
        elif self.counter >= self.buffer_size:
            self.make_curriculums()
            self.counter = 0
        self.counter += 1
        self.curriculum = int(self.counter // (self.buffer_size / self.number_of_curriculums)) % self.number_of_curriculums # The % is to make sure that the curriculum is always between 0 and number_of_curriculums
        index = np.random.choice(np.arange(len(self.buffer_deque_curriculum[self.curriculum])), size = batch_size, replace = False)
        return [self.buffer_deque_curriculum[self.curriculum][k] for k in index]

    def populate_memory_model(self, model, environment, name_env, k_initial_experiences, device):
        rewardbounds_per_env=pd.read_csv('experimenting/rl_utils/reward_data_per_environment.csv', delimiter=' ', header=0)
        min_r = rewardbounds_per_env[rewardbounds_per_env.Environment == name_env].Rminhard.item()
        max_r = rewardbounds_per_env[rewardbounds_per_env.Environment == name_env].Rmaxhard.item()
        normalize_reward = lambda r: (r - min_r) / (max_r - min_r)
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
                
                # Calculate temporal difference
                with torch.no_grad():
                    next_stacked_frames = torch.tensor([next_stacked_frames], device=device, dtype=torch.float32)
                    current_stacked_frames = torch.tensor([current_stacked_frames], device=device, dtype=torch.float32)
                    temporal_difference = ((reward + self.gamma * model.forward(next_stacked_frames).max(1)[0].item() - 
                                            model.forward(current_stacked_frames)[0,action])**2).item()

                # Add experience to replay memory
                self.add((current_stacked_frames, action, reward, next_stacked_frames, done, temporal_difference))
                
                # Start a new episode
                initial_frame = environment.reset()
                
                # Initialize stack
                stacked_frames_object.initialize_stack(initial_frame)
                
            else:
                stacked_frames_object.append_frame_to_stack(next_frame)
                next_stacked_frames = stacked_frames_object.stacked_frames_array

                # Calculate temporal difference
                with torch.no_grad():
                    next_stacked_frames = torch.tensor([next_stacked_frames], device=device, dtype=torch.float32)
                    current_stacked_frames = torch.tensor([current_stacked_frames], device=device, dtype=torch.float32)
                    temporal_difference = ((reward + self.gamma * model.forward(next_stacked_frames).max(1)[0].item() - 
                                            model.forward(current_stacked_frames)[0,action])**2).item()

                # Add experience to replay memory
                self.add((current_stacked_frames, action, reward, next_stacked_frames, done, temporal_difference))
    
    def make_curriculums(self):
        # Make curriculums based on temporal error
        td_list = [self.buffer_deque[i][-1] for i in range(len(self.buffer_deque))]
        td_list = np.array(td_list)
        # Get the indices to make self.number_of_curriculums curriculums
        curriculum_indices = np.array_split(np.argsort(td_list), self.number_of_curriculums)
        # Create sub dequeues for each curriculum
        self.buffer_deque_curriculum = [deque(maxlen = self.buffer_size // self.number_of_curriculums) for _ in range(self.number_of_curriculums)]
        for i, indices in enumerate(curriculum_indices):
            for index in indices:
                self.buffer_deque_curriculum[i].append(self.buffer_deque[index])

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
        rewardbounds_per_env=pd.read_csv('experimenting/rl_utils/reward_data_per_environment.csv', delimiter=' ', header=0)
        min_r = rewardbounds_per_env[rewardbounds_per_env.Environment == name_env].Rminhard.item()
        max_r = rewardbounds_per_env[rewardbounds_per_env.Environment == name_env].Rmaxhard.item()
        normalize_reward = lambda r: (r - min_r) / (max_r - min_r)
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