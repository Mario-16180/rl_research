import torch
import torch.nn as nn
import math
import random

class impala_cnn(nn.Module):
    """ This model was used in the paper "IMPALA: Scalable Distributed Deep-RL with
    Importance Weighted Actor-Learner Architectures" https://arxiv.org/abs/1802.01561 
    I only omit the LSTM part of the architecture, as I don't need it for this research."""
    def __init__(self, env, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.action_space = env.action_space.n
        self.build_network()

    def build_network(self):
        # First convolutional layer + pooling + block
        self.conv_pool_1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=1),
            nn.MaxPool2d(stride=[2, 2], kernel_size=[3, 3])
            )
        self.block_1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
        )

        # Second convolutional layer + pooling + block
        self.conv_pool_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
            nn.MaxPool2d(stride=[2, 2], kernel_size=[3, 3])
            )
        self.block_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
        )

        # Third convolutional layer + pooling + block
        self.conv_pool_3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.MaxPool2d(stride=[2, 2], kernel_size=[3, 3])
            )
        self.block_3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
        )

        # Flattening and MLP
        self.flatten_to_mlp = nn.Sequential(
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(800, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_space)
        )
    
    def forward(self, x):
        # Convert numpy array to torch tensor (1, 4, 64, 64)
        x = torch.Tensor(x)
        # Stack of 4 images to float32 and normalized
        x = x.float() / 255.0
        # First convolutional layer + pooling + block
        x = self.conv_pool_1(x)
        x_saved = x
        x = self.block_1(x) + x_saved
        # Second convolutional layer + pooling + block
        x = self.conv_pool_2(x)
        x_saved = x
        x = self.block_2(x) + x_saved
        # Third convolutional layer + pooling + block
        x = self.conv_pool_3(x)
        x_saved = x
        x = self.block_3(x) + x_saved
        # Flattening and MLP
        x = self.flatten_to_mlp(x)
        return x
    
    def save_model(self, episode, optimizer, loss, buffer, path):
        # Save the model
        torch.save({"episode": episode, "model_state_dict": self.state_dict(), "buffer": buffer, "optimizer_state_dict": optimizer.state_dict(), 
                    "loss": loss}, path)

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