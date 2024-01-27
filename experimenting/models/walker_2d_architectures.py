import torch
import torch.nn as nn
import math
import random
import pickle

"""
The architecture for the bipedal walker environment will consist of three networks since I will use soft actor critic.
The first network will be the actor, which will output the action to take.
The second network will be the critic, which will output the V value of the state.
The third network will be the Q network, which will output the Q value of the state-action pair.

Info about the bipedal walker environment:
- Observation space: Box(24,) (24 continuous values)
- Action space: Box(4,) (4 continuous values)
- Reward range: (-100, 300)

"""

# Define the Actor network
class actor_network(nn.Module):
    def __init__(self, lr, n_neurons_first_layer, n_neurons_second_layer, 
                 std_max, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.action_size = 4 # Fixed size for the action space in the walker2d environment
        self.state_dim = 24 # Fixed size for the state space in the walker2d environment
        self.n_neurons_first_layer = n_neurons_first_layer
        self.n_neurons_second_layer = n_neurons_second_layer
        self.std_min = 1e-6
        self.std_max = std_max

    def build_network(self):    
        self.network = nn.Sequential(
            nn.Linear(self.state_dim, self.n_neurons_first_layer),
            nn.ReLU(),
            nn.Linear(self.n_neurons_first_layer, self.n_neurons_second_layer),
            nn.ReLU(),
        )
        self.mean_fc = nn.Linear(self.n_neurons_second_layer, self.action_size)
        self.std_fc = nn.Linear(self.n_neurons_second_layer, self.action_size)

    def forward(self, state):
        x = self.network(state)
        mean = self.mean_fc(x)
        std = self.std_fc(x)
        std = torch.clamp(std, self.std_min, self.std_max)
        return mean, std
    
    def sample_action(self, state, device):
        mean, std = self.forward(state)
        normal = torch.distributions.Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z).detach().cpu().numpy()
        return action
    
    def sample(self, state):
        pass

# Define the Critic network
class critic_network(nn.Module):
    def __init__(self, lr, n_neurons_first_layer, n_neurons_second_layer, *args, **kwargs) -> None:
        super(critic_network, self).__init__()
        self.action_size = 4 # Fixed size for the action space in the walker2d environment
        self.state_dim = 24 # Fixed size for the state space in the walker2d environment
        self.n_neurons_first_layer = n_neurons_first_layer
        self.n_neurons_second_layer = n_neurons_second_layer
        self.build_network(n_neurons_first_layer, n_neurons_second_layer)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def build_network(self):
        self.network = nn.Sequential(
            nn.Linear(self.state_dim + self.action_size, self.n_neurons_first_layer),
            nn.ReLU(),
            nn.Linear(self.n_neurons_first_layer, self.n_neurons_second_layer),
            nn.ReLU(),
            nn.Linear(self.n_neurons_second_layer, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.network(x)
    
    def save_model(self, episode, train_step, optimizer, loss, buffer, path):
        # Save the model
        torch.save({"episode": episode, "step": train_step, "model_state_dict": self.state_dict(), "optimizer_state_dict": optimizer.state_dict(), 
                    "loss": loss}, path + ".tar")
        # Save the buffer
        pickle.dump(buffer, open(path + "_buffer", "wb"))
    
    def load_model(self, path):
        checkpoint = torch.load(path + ".tar")
        self.load_state_dict(checkpoint["model_state_dict"])
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        episode = checkpoint["episode"]
        train_step = checkpoint["step"]
        loss = checkpoint["loss"]
        buffer = pickle.load(open(path + "_buffer", "rb"))
        return episode, train_step, optimizer, loss, buffer

# Define the Q network
class q_network(nn.Module):
    def __init__(self, lr, n_neurons_first_layer, n_neurons_second_layer, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.action_size = 4 # Fixed size for the action space in the walker2d environment
        self.state_dim = 24 # Fixed size for the state space in the walker2d environment
        self.n_neurons_first_layer = n_neurons_first_layer
        self.n_neurons_second_layer = n_neurons_second_layer
        self.build_network(n_neurons_first_layer, n_neurons_second_layer)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def build_network(self):    
        self.network = nn.Sequential(
            nn.Linear(self.state_dim, self.n_neurons_first_layer),
            nn.ReLU(),
            nn.Linear(self.n_neurons_first_layer, self.n_neurons_second_layer),
            nn.ReLU(),
            nn.Linear(self.n_neurons_second_layer, 1)
        )

    def forward(self, state):
        x = self.network(state)
        return x
    
    def save_model(self, episode, train_step, optimizer, loss, buffer, path):
        # Save the model
        torch.save({"episode": episode, "step": train_step, "model_state_dict": self.state_dict(), "optimizer_state_dict": optimizer.state_dict(), 
                    "loss": loss}, path + ".tar")
        # Save the buffer
        pickle.dump(buffer, open(path + "_buffer", "wb"))
    
    def load_model(self, path):
        checkpoint = torch.load(path + ".tar")
        self.load_state_dict(checkpoint["model_state_dict"])
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        episode = checkpoint["episode"]
        train_step = checkpoint["step"]
        loss = checkpoint["loss"]
        buffer = pickle.load(open(path + "_buffer", "rb"))
        return episode, train_step, optimizer, loss, buffer