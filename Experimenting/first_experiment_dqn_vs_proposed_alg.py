import imageio
import os
import gym
import torch
import torch.nn as nn
import wandb

def save_frames_as_gif(frames, path='/home/mario.cantero/Documents/Research/rl_research/SavedRenders', filename='gym_animation.gif'):
    imageio.mimwrite(os.path.join(path, filename), frames, duration=20, format='gif')

class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen = max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size), size = batch_size, replace = False)
        return [self.buffer[i] for i in index]

    def populate_memory(self, pretrain_length, game):
        for i in range(pretrain_length):
            # First initializing transition
            if i == 0:
                stacked_frames_deque  =  deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=4)
                initial_frame = game.reset()
                stacked_frames, stacked_frames_deque = stack_frames(stacked_frames_deque, initial_frame, True, game)
                
            # Getting a random action from action space
            action = game.action_space.sample()
            next_frame, reward, done, _ = game.step(action)

            # Stack the frames
            next_stacked_frames, stacked_frames_deque = stack_frames(stacked_frames_deque, next_frame, False, game)
        
            # If it's the end of the episode
            if done:
                # End of episode
                next_stacked_frames = np.zeros(stacked_frames.shape)
                
                # Add experience to replay memory
                self.add((stacked_frames, action, reward, next_stacked_frames, done))
                
                # Start a new episode
                initial_frame = game.reset()
                
                # Stack the frames
                stacked_frames, stacked_frames_deque = stack_frames(stacked_frames_deque, initial_frame, True, game)
                
            else:
                # Add experience to replay memory
                self.add((stacked_frames, action, reward, next_stacked_frames, done))
                
                # Update current state
                stacked_frames = next_stacked_frames

class q_network(nn.Module):
    def __init__(self, path='/home/mario.cantero/Documents/Research/rl_research/Experimenting/Models'):
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_path = path

    def build_model(self):
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
    "dataset": "procgen",
    "epochs": 1000}
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