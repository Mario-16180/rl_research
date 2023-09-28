import imageio
import os
import gym
import torch
import torch.nn as nn
import wandb

def save_frames_as_gif(frames, path='/home/mario.cantero/Documents/Research/rl_research/SavedRenders', filename='gym_animation.gif'):
    imageio.mimwrite(os.path.join(path, filename), frames, duration=20, format='gif')

class q_network(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x / 255.0)

def train_new_cl_algo():
    pass

def train_vanilla_dqn(env, model):
    pass

if __name__ == '__main__':

    # start a new wandb run to track this script
    wandb.init(
    # set the wandb project where this run will be logged
    project="rl_research_mbzuai",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.02,
    "architecture": "CNN",
    "dataset": "CIFAR-100",
    "epochs": 10}
    )

    env_name = "procgen:procgen-bossfight-v0"
    env = gym.make(env_name, render_mode="rgb_array")
    obs = env.reset()
    frames = []
    iter = 0
    while True:
        obs, rew, done, info = env.step(env.action_space.sample())
        print(rew, 'step: ', iter)
        iter += 1
        #frame = env.render()
        frames.append(obs)
        if done:
            break
    save_frames_as_gif(frames, filename='pruebita2.gif')