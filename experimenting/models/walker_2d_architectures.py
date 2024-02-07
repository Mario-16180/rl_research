import torch
import torch.nn as nn
import pickle
from torch.distributions.normal import Normal

"""
The architecture for the bipedal walker environment will consist of three networks since I will use soft actor critic.
The first network will be the actor, which will output the action to take.
The second network will be the critic, which will output the V value of the state.
The third network will be the Q network, which will output the Q value of the state-action pair.

Info about the bipedal walker environment:
- Observation space: Box(24,) (24 continuous values)
- Action space: Box(4,) (4 continuous values, from -1 to 1, so no need to scale the actions)
- Reward range: (-100, 300)

"""

# Define the Actor network
class actor_network(nn.Module):
    def __init__(self, lr, n_neurons_first_layer, n_neurons_second_layer, device, std_max=2) -> None:
        super(actor_network, self).__init__()
        self.action_size = 4 # Fixed size for the action space in the walker2d environment
        self.state_dim = 24 # Fixed size for the state space in the walker2d environment
        self.n_neurons_first_layer = n_neurons_first_layer
        self.n_neurons_second_layer = n_neurons_second_layer
        self.device = device
        self.std_min = 1e-6
        self.std_max = std_max
        self.build_network()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)

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
    
    def sample_action(self, state, reparameterize=True):
        mean, std = self.forward(state)
        normal = Normal(mean, std)
        if reparameterize:
            z = normal.rsample()
        else:
            z = normal.sample()
        action = torch.tanh(z).to(self.device) # Range from -1 to 1, so we don't need to scale the actions since the action space is already from -1 to 1
        log_prob = normal.log_prob(z) # Taking the log of the probability density function of the normal distribution
        log_prob -= torch.log(torch.tensor(1 - action**2 + self.std_min, device=self.device, dtype=torch.float32))
        log_prob = log_prob.sum(1, keepdim=True) # Since the Jacobian of tanh is 1 - tanh(x)^2 and diagonal, we need to sum the log_probabilities
        return action, log_prob
    
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

# Define the Critic network (Q function)
class critic_network(nn.Module):
    def __init__(self, lr, n_neurons_first_layer, n_neurons_second_layer, device) -> None:
        super(critic_network, self).__init__()
        self.action_size = 4 # Fixed size for the action space in the walker2d environment
        self.state_dim = 24 # Fixed size for the state space in the walker2d environment
        self.device = device
        self.n_neurons_first_layer = n_neurons_first_layer
        self.n_neurons_second_layer = n_neurons_second_layer
        self.build_network()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)

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

# Define the Value network (V function)
class v_network(nn.Module):
    def __init__(self, lr, n_neurons_first_layer, n_neurons_second_layer, device) -> None:
        super(v_network, self).__init__()
        self.action_size = 4 # Fixed size for the action space in the walker2d environment
        self.state_dim = 24 # Fixed size for the state space in the walker2d environment
        self.device = device
        self.n_neurons_first_layer = n_neurons_first_layer
        self.n_neurons_second_layer = n_neurons_second_layer
        self.build_network()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)

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