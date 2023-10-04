import torch
import torch.nn as nn

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
            nn.Flatten(start_dim=0),
            nn.Linear(800, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_space)
        )
    
    def forward(self, x):
        # Convert numpy array of shape (4, 64, 64) to torch tensor of shape (1, 4, 64, 64)
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
    
    def save_model(self):
        pass

    def load_model(self):
        pass

"""
if __name__ == '__main__':
    env_name = "procgen:procgen-bossfight-v0"
    env = gym.make(env_name)
    print(env.action_space.sample(), env.action_space.n)
    obs = env.reset()
 
    model = impala_cnn(env)
    model.build_network()
    print(model)
    x = torch.rand((10, 4, 64, 64))
    print(model(x).shape)
    print(model(x))
"""
