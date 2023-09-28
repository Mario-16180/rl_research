import imageio
import os
import gym
import torch
import wandb

def save_frames_as_gif(frames, path='/home/mario.cantero/Documents/Research/rl_research/SavedRenders', filename='gym_animation.gif'):
    imageio.mimwrite(os.path.join(path, filename), frames, duration=20, format='gif')

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