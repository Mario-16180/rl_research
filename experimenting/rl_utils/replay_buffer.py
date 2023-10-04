from collections import deque
from stack_frames import stacked_frames_class
import numpy as np
import gym

class memory():
    def __init__(self, max_size):
        self.buffer_size = max_size
        self.buffer_deque = deque(maxlen = max_size)
    
    def add(self, experience):
        self.buffer_deque.append(experience)
    
    def sample(self, batch_size):
        index = np.random.choice(np.arange(self.buffer_size), size = batch_size, replace = False)
        return [self.buffer_deque[i] for i in index]

    def populate_memory_random(self, environment, k_initial_experiences):
        for i in range(k_initial_experiences):
            # First initializing transition
            if i == 0:
                stacked_frames_object = stacked_frames_class()
                initial_frame = environment.reset()
                stacked_frames_object.initialize_stack(initial_frame)
                
            # Getting a random action from action space
            action = environment.action_space.sample()
            next_frame, reward, done, _ = environment.step(action)

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

if __name__ == '__main__':
    env = gym.make('procgen:procgen-bossfight-v0', num_levels=1, start_level=0)
    memory_object = memory(max_size=10000)
    memory_object.populate_memory_random(env, k_initial_experiences=10000)
    print(len(memory_object.buffer_deque))