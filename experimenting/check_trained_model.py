# Check trained model
import torch
import torch.nn as nn
import math
import random
import pickle
import gym
import numpy as np
from rl_utils.stack_frames import stacked_frames_class
from rl_utils.saving_frames import save_frames_as_gif
from models.impala_cnn_architecture import impala_cnn

path = "experimenting/models/trained_models/golden-armadillo-20procgen:procgen-bossfight-v0.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make("procgen:procgen-bossfight-v0")
state = env.reset()
stacked_frames = stacked_frames_class()
stacked_frames.initialize_stack(state)

# Read model from pt file
file = torch.load(path, map_location=device)
model = impala_cnn(env).to(device)
model.load_state_dict(file)

done = False
reward_accumulated = 0
# Initialize frame array
frames_accumulated = []
while not done:
    action, _ = model.select_action(env, stacked_frames.stacked_frames_array, 0.01, 1, 0.01, 0, device)
    action = action.item()
    state, reward, done, info = env.step(action)
    
    stacked_frames.append_frame_to_stack(state)
    env.render()
    reward_accumulated += reward
    print(reward_accumulated)
    frames_accumulated.append(env.render(mode="rgb_array"))
    if done:
        break
save_frames_as_gif(frames_accumulated, path="experimenting/saved_renders", filename="trained_model_check.gif")
env.close()