import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device= "cude" if torch.cuda.is_available() else "cpu"

class BaseNeuralNetwork(nn.Module):
    def __init__(self):
        super(BaseNeuralNetwork, self).__init__()
        self.flatten= nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(300 * 300 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )
    
    def forward(self, x):
        x= self.flatten(x)
        logits= self.linear_relu_stack(x)
        return logits

def train_loop(dataloader, model, loss_fn, optimizer):
    """Trains PyTorch model using training dataloader with specified
    loss function and optimizer.
    """
    size= len(dataloader.dataset)

    for batch, (X, y) in enumerate(dataloader):
        predictions= model(X.float())
        loss= loss_fn(predictions, y.flatten())

        #Reset gradient 
        optimizer.zero_grad()

        #Calculate Gradient
        loss.backward()

        #update weights
        optimizer.step()

        
        loss, current = loss.item(), batch * len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]") 

def 

