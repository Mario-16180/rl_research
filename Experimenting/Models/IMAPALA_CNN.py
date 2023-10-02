import torch
import torch.nn as nn

class impala_cnn(nn.Module):
    """ This model was used in the paper "IMPALA: Scalable Distributed Deep-RL with
    Importance Weighted Actor-Learner Architectures" https://arxiv.org/abs/1802.01561 
    I only omit the LSTM part of the architecture, as I don't need it for this research."""
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def build_network(self):
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