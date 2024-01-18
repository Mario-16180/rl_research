from collections import deque
from rl_utils.stack_frames import stacked_frames_class
import numpy as np
import torch
import pandas as pd

class memory_with_curriculum():
    def __init__(self, max_size, max_train_steps_per_curriculum_criterion1, max_train_steps_per_curriculum_criterion2, stability_dequeue_size=2500, gamma=0.99, curriculums=3):
        self.buffer_size = max_size
        self.buffer_deque = deque(maxlen = max_size)
        self.losses_deque = deque(maxlen = stability_dequeue_size) # For the stability criterion
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.number_of_curriculums = curriculums
        self.flag_first_curriculum = True
        self.counter = 0
        self.max_train_per_curriculum_criterion1 = max_train_steps_per_curriculum_criterion1
        self.max_train_per_curriculum_criterion2 = max_train_steps_per_curriculum_criterion2
    
    def add(self, experience):
        self.buffer_deque.append(experience)
    
    def sample(self, batch_size, curriculum_criterion=1, anti_curriculum=False):
        # Criteria for getting to the next curriculum and therefore affecting the sampling. Criterion number 1 corresponds to changing curriculums based on the
        # amount of new experiences in the current curriculum. Criterion number 2 corresponds to stability of the loss function.
        if curriculum_criterion == 1:
            if self.flag_first_curriculum:
                self.make_curriculums(anti_curriculum=anti_curriculum)
                self.flag_first_curriculum = False
                self.curriculum = 0
            elif self.counter >= self.max_train_per_curriculum_criterion1:
                self.curriculum += 1
                self.counter = 0
            if self.curriculum >= self.number_of_curriculums:
                self.make_curriculums(anti_curriculum=anti_curriculum)
                self.curriculum = 0
            self.counter += 1
            index = np.random.choice(np.arange(len(self.buffer_deque_curriculum[self.curriculum])), size = batch_size, replace = False)
            return [self.buffer_deque_curriculum[self.curriculum][k] for k in index]
        elif curriculum_criterion == 2:
            if self.flag_first_curriculum:
                self.make_curriculums(anti_curriculum=anti_curriculum)
                self.flag_first_curriculum = False
                self.curriculum = 0
            elif len(self.losses_deque) == self.losses_deque.maxlen:
                """
                To know if the curve has already hit a plateau, I'm gonna take the proportion between the average of the first 50 elements and the average of the last 50 elements.
                If it's less than 1.5 (which corresponds to 150 % of proportion), then it's a plateau and we can continue to the next curriculum.
                """
                first_50 = np.mean(list(self.losses_deque)[:50])
                last_50 = np.mean(list(self.losses_deque)[-50:])
                if (last_50 / first_50 < 1.5) or self.counter >= self.max_train_per_curriculum_criterion2:
                    self.losses_deque.clear()
                    self.curriculum += 1
                    self.counter = 0
            # Make another set of curricula if the last curriculum has already been trained on.
            if self.curriculum >= self.number_of_curriculums:
                self.make_curriculums(anti_curriculum=anti_curriculum)
                self.curriculum = 0
                self.counter = 0
            self.counter += 1
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
                # Convert tensors to numpy arrays
                next_stacked_frames = next_stacked_frames.cpu().numpy()
                current_stacked_frames = current_stacked_frames.cpu().numpy()
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
                # Convert tensors to numpy arrays
                next_stacked_frames = next_stacked_frames.cpu().numpy()
                current_stacked_frames = current_stacked_frames.cpu().numpy()
                # Add experience to replay memory
                self.add((current_stacked_frames, action, reward, next_stacked_frames, done, temporal_difference))
    
    def make_curriculums(self, anti_curriculum=False):
        if anti_curriculum:
            # Make curriculums based on temporal error
            td_list = [self.buffer_deque[i][-1] for i in range(len(self.buffer_deque))]
            td_list = np.array(td_list)
            # Get the indices to make self.number_of_curriculums curriculums
            curriculum_indices = np.array_split(np.argsort(td_list)[::-1], self.number_of_curriculums)
            # Create sub dequeues for each curriculum
            self.buffer_deque_curriculum = [deque(maxlen = self.buffer_size // self.number_of_curriculums) for _ in range(self.number_of_curriculums)]
            for i, indices in enumerate(curriculum_indices):
                for index in indices:
                    self.buffer_deque_curriculum[i].append(self.buffer_deque[index])
        else:
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

class memory_lunar_lander():
    def __init__(self, max_size):
        self.buffer_size = max_size
        self.buffer_deque = deque(maxlen = max_size)

    def add(self, experience):
        self.buffer_deque.append(experience)
    
    def sample(self, batch_size):
        index = np.random.choice(np.arange(len(self.buffer_deque)), size = batch_size, replace = False)
        return [self.buffer_deque[i] for i in index]

    def populate_memory_random(self, environment, k_initial_experiences):
        done = True
        for i in range(k_initial_experiences):
            if done:
                current_state = environment.reset()[0]
            # Getting a random action from action space
            action = environment.action_space.sample()
            next_state, reward, done, _, info = environment.step(action)
            self.add((current_state, action, reward, next_state, done))
            current_state = next_state

class memory_lunar_lander_curriculum():
    def __init__(self, max_size, max_train_steps_per_curriculum_criterion1, max_train_steps_per_curriculum_criterion2, 
                 stability_dequeue_size=2500, gamma=0.99, curriculums=3, anti_curriculum=False, percentile=0.5):
        self.buffer_size = max_size
        self.buffer_deque = deque(maxlen = max_size)
        self.losses_deque = deque(maxlen = stability_dequeue_size) # For the stability criterion
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.number_of_curriculums = curriculums
        self.flag_first_curriculum = True
        self.counter = 0
        self.max_train_per_curriculum_criterion1 = max_train_steps_per_curriculum_criterion1
        self.max_train_per_curriculum_criterion2 = max_train_steps_per_curriculum_criterion2
        self.percentile = percentile
        self.anti_curriculum = anti_curriculum
    
    def add(self, experience):
        self.buffer_deque.append(experience)
    
    def sample(self, batch_size, curriculum_criterion=1, anti_curriculum=False):
        # Criteria for getting to the next curriculum and therefore affecting the sampling. Criterion number 1 corresponds to changing curriculums based on the
        # amount of new experiences made with the current curriculum. Criterion number 2 corresponds to stability of the loss function.
        if curriculum_criterion == 1:
            if self.flag_first_curriculum:
                self.make_curriculums()
                self.flag_first_curriculum = False
                self.curriculum = 0
            elif self.counter >= self.max_train_per_curriculum_criterion1:
                self.curriculum += 1
                self.counter = 0
            if self.curriculum >= self.number_of_curriculums:
                self.make_curriculums()
                self.curriculum = 0
            self.counter += 1
            index = np.random.choice(np.arange(len(self.buffer_deque_curriculum[self.curriculum])), size = batch_size, replace = False)
            return [self.buffer_deque_curriculum[self.curriculum][k] for k in index]
        # Currently, criterion 2 is not being used
        elif curriculum_criterion == 2:
            if self.flag_first_curriculum:
                self.make_curriculums() 
                self.flag_first_curriculum = False
                self.curriculum = 0
            elif len(self.losses_deque) == self.losses_deque.maxlen:
                """
                To know if the curve has already hit a plateau, I'm gonna take the proportion between the average of the first 50 elements and the average of the last 50 elements.
                If it's less than 1.5 (which corresponds to 150 % of proportion), then it's a plateau and we can continue to the next curriculum.
                """
                first_50 = np.mean(list(self.losses_deque)[:50])
                last_50 = np.mean(list(self.losses_deque)[-50:])
                if (last_50 / first_50 < 1.5) or self.counter >= self.max_train_per_curriculum_criterion2:
                    self.losses_deque.clear()
                    self.curriculum += 1
                    self.counter = 0
            # Make another set of curricula if the last curriculum has already been trained on.
            if self.curriculum >= self.number_of_curriculums:
                self.make_curriculums()
                self.curriculum = 0
                self.counter = 0
            self.counter += 1
            index = np.random.choice(np.arange(len(self.buffer_deque_curriculum[self.curriculum])), size = batch_size, replace = False)
            return [self.buffer_deque_curriculum[self.curriculum][k] for k in index]

    def populate_memory_random(self, model, environment, k_initial_experiences, device):
        done = True
        for i in range(k_initial_experiences):
            if done:
                current_state = environment.reset()[0]
            # Getting a random action from action space
            action = environment.action_space.sample()
            next_state, reward, done, _, info = environment.step(action)
            with torch.no_grad():
                    next_state = torch.tensor([next_state], device=device, dtype=torch.float32)
                    current_state = torch.tensor([current_state], device=device, dtype=torch.float32).reshape(1,8)
                    temporal_difference = ((reward + self.gamma * model.forward(next_state).max(1)[0].item() - 
                                            model.forward(current_state)[0,action])**2).item()
            next_state = next_state.cpu().numpy()
            current_state = current_state.cpu().numpy()
            self.add((current_state, action, reward, next_state, done, temporal_difference))
            current_state = next_state

    def make_curriculums(self):
        # Make curriculums based on temporal error
        td_list = [self.buffer_deque[i][-1] for i in range(len(self.buffer_deque))]
        td_list = np.array(td_list[int(len(td_list) * self.percentile / 100):])
        if self.anti_curriculum:
            # Get the indices to make self.number_of_curriculums curriculums
            curriculum_indices = np.array_split(np.argsort(td_list)[::-1], self.number_of_curriculums)
        else:
            # Get the indices to make self.number_of_curriculums curriculums
            curriculum_indices = np.array_split(np.argsort(td_list), self.number_of_curriculums)
        # Create sub dequeues for each curriculum
        self.buffer_deque_curriculum = [deque(maxlen = self.buffer_size // self.number_of_curriculums) for _ in range(self.number_of_curriculums)]
        for i, indices in enumerate(curriculum_indices):
            for index in indices:
                self.buffer_deque_curriculum[i].append(self.buffer_deque[index])