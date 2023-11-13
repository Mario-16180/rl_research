import torch
import torch.nn as nn
import math
import random
import pickle

class lunar_lander_cnn(nn.Module):
    def __init__(self, env, n_neurons_first_layer, n_neurons_second_layer, *args, **kwargs) -> None:
        """
        Build a fully connected neural network
        
        Parameters
        ----------
        self.observation_size (int): Observation dimension
        self.action_size (int): Action dimension
        """
        super().__init__(*args, **kwargs)
        self.action_size = env.action_space.n
        self.observation_size = env.observation_space.shape[0]
        self.build_network(n_neurons_first_layer, n_neurons_second_layer)

    def build_network(self, n_neurons_first_layer, n_neurons_second_layer):
        # Nature CNN architecture
        self.conv_pool = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            )
        self.flatten_to_mlp = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(3136, n_neurons_first_layer),
            nn.ReLU(),
            nn.Linear(n_neurons_first_layer, n_neurons_second_layer),
            nn.ReLU(),
            nn.Linear(n_neurons_second_layer, self.action_size)
        )

    
    def forward(self, x):
        # Convert numpy array to torch tensor (1, 4, 64, 64)
        x = torch.Tensor(x)
        # Stack of 4 images to float32 and normalized
        x = x.float() / 255.0
        # First convolutional layer + pooling + block
        x = self.conv_pool(x)
        # Flattening and MLP
        x = self.flatten_to_mlp(x)
        return x
    
    def save_model(self, episode, train_step, optimizer, loss, buffer, path):
        # Save the model
        torch.save({"episode": episode, "step": train_step, "model_state_dict": self.state_dict(), "optimizer_state_dict": optimizer.state_dict(), 
                    "loss": loss}, path + ".tar")
        # Save the buffer
        pickle.dump(buffer, open(path + "_buffer", "wb"))

    def select_action(self, env, state, epsilon_start, epsilon_decay, epsilon_min, current_step, device):
        sample_for_probability = random.random()
        eps_threshold = epsilon_min + (epsilon_start - epsilon_min) * math.exp(-1. * current_step / epsilon_decay)
        if sample_for_probability > eps_threshold:
            with torch.no_grad():
                state = torch.tensor([state], device=device, dtype=torch.float32)
                action = self.forward(state).max(1)[1].view(1, 1)
                return action, eps_threshold
        else:
            action = torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
            return action, eps_threshold