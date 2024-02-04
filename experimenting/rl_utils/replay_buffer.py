from collections import deque
import numpy as np
import torch

class memory_lunar_lander():
    def __init__(self, buffer_size, max_train_steps_per_curriculum=1000, use_curriculum=False, n_curriculums=3, anti_curriculum=False, percentile=0):
        # Parameters of a plain replay buffer
        self.buffer_size = buffer_size
        self.buffer_deque = deque(maxlen = self.buffer_size)
        # PArameters of a curriculum replay buffer
        self.max_train_per_curriculum = max_train_steps_per_curriculum
        self.use_curriculum = use_curriculum
        self.n_curriculums = n_curriculums
        self.anti_curriculum = anti_curriculum
        self.percentile = percentile
        # Miscellaneous parameters. Counter is used to keep track of the amount of training steps made with the current curriculum.
        self.counter = 0
        self.flag_first_curriculum = True
        
    
    def add(self, experience):
        self.buffer_deque.append(experience)
    
    def sample(self, batch_size):
        # Criteria for getting to the next curriculum and therefore affecting the sampling. Criterion number 1 corresponds to changing curriculums based on the
        # amount of new experiences made with the current curriculum. Criterion number 2 corresponds to stability of the loss function.
        if self.use_curriculum:
            if self.flag_first_curriculum:
                self.make_curriculums()
                self.flag_first_curriculum = False
                self.curriculum = 0
            elif self.counter >= self.max_train_per_curriculum:
                self.curriculum += 1
                self.counter = 0
            if self.curriculum >= self.n_curriculums:
                self.make_curriculums()
                self.curriculum = 0
            self.counter += 1
            index = np.random.choice(np.arange(len(self.buffer_deque_curriculum[self.curriculum])), size = batch_size, replace = False)
            return [self.buffer_deque_curriculum[self.curriculum][k] for k in index]
        else:
            index = np.random.choice(np.arange(len(self.buffer_deque)), size = batch_size, replace = False)
            return [self.buffer_deque[i] for i in index]

    def populate_memory_random(self, environment, k_initial_experiences, gamma, model, device):
        done = True
        for _ in range(k_initial_experiences):
            if done:
                current_state = environment.reset()[0]
            # Getting a random action from action space
            action = environment.action_space.sample()
            next_state, reward, done, _, info = environment.step(action)
            if self.curriculum:
                with torch.no_grad():
                        next_state = torch.tensor([next_state], device=device, dtype=torch.float32)
                        current_state = torch.tensor([current_state], device=device, dtype=torch.float32).reshape(1,8)
                        temporal_difference = ((reward + gamma * model.forward(next_state).max(1)[0].item() - 
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
            curriculum_indices = np.array_split(np.argsort(td_list)[::-1], self.n_curriculums)
        else:
            # Get the indices to make self.number_of_curriculums curriculums
            curriculum_indices = np.array_split(np.argsort(td_list), self.n_curriculums)
        # Create sub dequeues for each curriculum
        self.buffer_deque_curriculum = [deque(maxlen = self.buffer_size // self.n_curriculums) for _ in range(self.n_curriculums)]
        for i, indices in enumerate(curriculum_indices):
            for index in indices:
                self.buffer_deque_curriculum[i].append(self.buffer_deque[index])

class memory_bipedal_walker():
    def __init__(self, buffer_size, use_curriculum=False, max_steps_per_curriculum=1000, n_curriculums=5, 
                 anti_curriculum=False, percentile=0):
        self.buffer_size = buffer_size
        self.buffer_deque = deque(maxlen = self.buffer_size)
        self.use_curriculum = use_curriculum
        self.max_steps_per_curriculum = max_steps_per_curriculum
        self.n_curriculums = n_curriculums
        self.anti_curriculum = anti_curriculum
        self.percentile = percentile
        self.counter = 0
        self.flag_first_curriculum = True

    def add(self, experience):
        """
        The experience transition consists of a tuple of 5 if curriculum learning is not used, but 6 if it is used.
        The first 5 elements are the following:
        - current_state: The current state of the environment. As a numpy array.
        - action: The action taken by the agent. As a numpy array.
        - reward: The reward obtained from the environment. As a scalar.
        - next_state: The next state of the environment. As a numpy array.
        - done: Whether the episode has ended or not. As a boolean.
        The last element is the temporal difference error, which is only used if curriculum learning is used.
        """
        self.buffer_deque.append(experience)

    def sample(self, batch_size):
        if self.use_curriculum:
            if self.flag_first_curriculum:
                self.make_curriculums()
                self.flag_first_curriculum = False
                self.current_curriculum = 0
            elif self.counter >= self.max_steps_per_curriculum:
                self.current_curriculum += 1
                self.counter = 0
            if self.current_curriculum >= self.n_curriculums:
                self.make_curriculums()
                self.current_curriculum = 0
            self.counter += 1
            indices = np.random.choice(np.arange(len(self.buffer_deque_curriculum[self.current_curriculum])), size=batch_size, replace=False)
            return [self.buffer_deque_curriculum[self.current_curriculum][k] for k in indices]
        else:
            indices = np.random.choice(np.arange(len(self.buffer_deque)), size = batch_size, replace=False)
            return [self.buffer_deque[i] for i in indices]
        
    def make_curriculums(self):
        # Make curriculums based on temporal error
        td_list = [self.buffer_deque[i][-1] for i in range(len(self.buffer_deque))]
        td_list = np.array(td_list[int(len(td_list) * self.percentile / 100):])
        if self.anti_curriculum:
            # Get the indices to n curriculums
            curriculum_indices = np.array_split(np.argsort(td_list)[::-1], self.n_curriculums)
        else:
            # Get the indices to make n curriculums
            curriculum_indices = np.array_split(np.argsort(td_list), self.n_curriculums)
        # Create sub dequeues for each curriculum
        self.buffer_deque_curriculum = [deque(maxlen = self.buffer_size // self.n_curriculums) for _ in range(self.n_curriculums)]
        for i, indices in enumerate(curriculum_indices):
            for index in indices:
                self.buffer_deque_curriculum[i].append(self.buffer_deque[index])