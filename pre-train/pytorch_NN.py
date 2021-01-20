
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import sys
import math


class Net(nn.Module):
    def __init__(self, width):
        self.width = width
        super(Net, self).__init__()
        # self.layer1 = nn.Sequential(nn.Linear(1, width), nn.ReLU(True))
        self.fc1 = nn.Linear(1, width) 
        self.fc2 = nn.Linear(width, 1)
        # self.add_module("fc1", self.layer1)
        # self.add_module("fc2", self.layer2)
        nn.init.uniform_(self.fc1.weight, 0, 1.0 / width)
        nn.init.uniform_(self.fc1.bias, 0, 1.0 / width)
        nn.init.uniform_(self.fc2.weight, 0, 1.0 / width)
        nn.init.uniform_(self.fc2.bias, 0, 1.0)
        # print(self._modules)
        
    def forward(self, x):

        with torch.no_grad():
            self.fc1.weight.masked_scatter_(self.fc1.weight != F.relu(self.fc1.weight), F.relu(self.fc1.weight))
            self.fc2.weight.masked_scatter_(self.fc2.weight != F.relu(self.fc2.weight), F.relu(self.fc2.weight))

        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        return out

    def train(self, x, y):
        optimizer = torch.optim.Adam(self.parameters(), lr = 0.001)
        loss_func = nn.L1Loss()
        
        for t in range(2000):
            # prediction = self.forward(x.float())
            prediction = self.forward(x.double())
            # print(prediction)
            # loss = loss_func(prediction, y.float())
            loss = loss_func(prediction, y.double())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if t % 100 == 0:
            #     print("--------------loss-----------: ", loss)

        final_prediction = self.forward(x.double())
        old_pred = sys.float_info.min
        for pred in final_prediction:
            if pred < old_pred:
                pred
            else:
                old_pred = pred
        # print(self.fc1.weight)
        # print(self.fc1.bias)
        # print(self.fc2.weight)
        # print(self.fc2.bias)

if __name__ == "__main__":
    net = Net(50)