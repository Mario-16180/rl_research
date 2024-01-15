import torch
import torch.nn as nn
import math
import random
import pickle

class lunar_lander_mlp(nn.Module):
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
        self.mlp = nn.Sequential(
            nn.Linear(self.observation_size, n_neurons_first_layer),
            nn.ReLU(),
            nn.Linear(n_neurons_first_layer, n_neurons_second_layer),
            nn.ReLU(),
            nn.Linear(n_neurons_second_layer, self.action_size)
        )
    
    def forward(self, x):
        x = torch.Tensor(x)
        return self.mlp(x)
    
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
                state = torch.tensor([state], device=device, dtype=torch.float32).reshape(1,8)
                action = self.forward(state).max(1)[1].view(1, 1)
                # action = self.forward(state).max(1)[1].view(1, 1)
                action = action.item()
                return action, eps_threshold
        else:
            action = env.action_space.sample()
            return action, eps_threshold