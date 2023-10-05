from collections import deque
import numpy as np

class stacked_frames_class():
    def __init__(self, stack_size=4) -> None:
        self.stack_size = stack_size

    def initialize_stack(self, initial_frame):
        self.stacked_frames_deque = deque([np.zeros((64,64), dtype=np.uint8) for _ in range(self.stack_size)], maxlen=self.stack_size)
        initial_frame = self.preprocess_new_frame(initial_frame)
        for _ in range(self.stack_size):
            self.stacked_frames_deque.append(initial_frame)
        self.stacked_frames_array = np.stack(self.stacked_frames_deque, axis=0)

    def append_frame_to_stack(self, frame):
        frame = self.preprocess_new_frame(frame)
        self.stacked_frames_deque.append(frame)
        self.stacked_frames_array = np.stack(self.stacked_frames_deque, axis=0)

    def preprocess_new_frame(self, frame):
        frame = np.dot(frame[...,:3], [0.2989, 0.5870, 0.1140])
        frame = frame.astype(np.uint8)
        return frame

"""
if __name__ == '__main__':
    env_name = "procgen:procgen-bossfight-v0"
    env = gym.make(env_name, num_levels=1, start_level=0)
    obs = env.reset()
    stacked_frames = stacked_frames_class()
    stacked_frames.initialize_stack(obs)
    print(stacked_frames.stacked_frames_array, stacked_frames.stacked_frames_array.shape)
    stacked_frames.append_frame_to_stack(obs)
    print(stacked_frames.stacked_frames_array, stacked_frames.stacked_frames_array.shape)
"""