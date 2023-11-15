import torch
import torch.nn as nn
import math
import random
import pickle

"""
The architecture for the walker2d environment will consist of two networks since I will use soft actor critic.
The first network will be the actor, which will output the action to take.
The second network will be the critic, which will output the V value of the state.
The third network will be the Q network, which will output the Q value of the state-action pair.

Info about the MuJoCo walker2d environment:
- Observation space: Box(17,) (17 continuous values)
- Action space: Box(6,) (6 continuous values)
- Reward range: (-inf, inf)

"""

# Define the Actor network
class actor_network(nn.Module):
    def __init__(self, env, hidden_size=256, log_std_min=-20, log_std_max=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_dim = env.action_space.shape[0]
        self.observation_size = env.observation_space.shape[0]
        self.hidden_size = hidden_size
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def build_network(self):    
        self.network = nn.Sequential(
            nn.Linear(self.observation_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
        )
        self.mean_fc = nn.Linear(self.hidden_size, self.action_dim)
        self.log_std_fc = nn.Linear(self.hidden_size, self.action_dim)

    def forward(self, state):
        x = self.network(state)
        mean = self.mean_fc(x)
        log_std = self.log_std_fc(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def sample(self, state):
        pass

# Define the Critic network
class critic_network(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(critic_network, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Define the Q network
class q_network(nn.Module):
    def __init__(self, env, hidden_size=256, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_dim = env.action_space.shape[0]
        self.observation_size = env.observation_space.shape[0]
        self.hidden_size = hidden_size

    def build_network(self):    
        self.network = nn.Sequential(
            nn.Linear(self.observation_size + self.action_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.network(x)