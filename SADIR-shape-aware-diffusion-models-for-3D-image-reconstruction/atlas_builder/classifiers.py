import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class simpleeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 5, stride=1, padding = 1, bias = True)
        nn.init.xavier_uniform_(self.conv1.weight)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 5, stride=1, padding = 1, bias = True)
        nn.init.xavier_uniform_(self.conv2.weight)
        self.conv3 = nn.Conv2d(16, 8, 5, stride=1, padding = 1, bias = True)
        nn.init.xavier_uniform_(self.conv3.weight)
        self.fc1 = nn.Linear(1568, 2048) 
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(2048, 200)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc3 = nn.Linear(200, 2)
        nn.init.xavier_uniform_(self.fc3.weight)
        self.act = nn.PReLU()

    def forward(self, x, latent ):
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = self.pool(self.act(self.conv3(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.act(self.fc1(x))
        latent = torch.flatten(latent, 1) 
        # print (latent.shape)
        x = x + latent 
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return x