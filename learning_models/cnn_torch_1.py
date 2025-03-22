import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self, in_channels, vis_side, *args, **kwargs):
        super().__init__(*args, **kwargs)
        conv_layers = [
            nn.Conv2d(min(*[64, in_channels * (2 ** i)]), min(*[64, in_channels * (2 ** (i + 1))]), 3)\
                for i in range((vis_side // 2) - 1)
        ]

        activation = nn.ReLU

        self.conv_layers = nn.Sequential()
        for i, layer in enumerate(conv_layers):
            self.conv_layers.add_module(f"conv_{i}", layer)
            self.conv_layers.add_module(f"actv_{i}", activation())

        out = nn.Softmax(dim=0)
        self.linear_layers = nn.Sequential(
            nn.Linear(9 * 64, 64),
            activation(),
            nn.Linear(64, 32),
            activation(),
            nn.Linear(32, 32),
            activation(),
            nn.Linear(32, 4),
            out
        )
        
    def forward(self, x):
        x = torch.permute(x, (2, 0, 1))
        x = self.conv_layers(x)
        x = torch.flatten(x)
        x = self.linear_layers(x)

        return x


# print(torch.argmax(CNN(8, 11)(torch.rand((11, 11, 8)))).item())