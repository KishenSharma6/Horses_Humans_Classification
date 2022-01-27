import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device= "cude" if torch.cuda.is_available() else "cpu"

class BaseNeuralNetwork(nn.Module):
    def __init__(self):
        super(BaseNeuralNetwork, self).__init__()
        self.flatten= nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(300 * 300, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )
    
    def forward(self, x):
        x= self.flatten(x)
        logits= self.linear_relu_stack(x)
        return logits