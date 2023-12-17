import os
import sys
from omegaconf import OmegaConf
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as F

from rabetorch.builders.solver_builder import SolverBuilder


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.fc1 = nn.Linear(256 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 256 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def test_build_sgd():
    # set config
    cfg_dict = {
        "OPTIMIZER": "SGD",
        "LOSS": {
            "TYPE": "CrossEntropyLoss",
        },
        "BASE_LR": 0.01,
        "MAX_EPOCH": 10,
    }
    cfg = OmegaConf.create(cfg_dict)

    # build solver
    model = Net()
    solver = SolverBuilder(cfg)
    solver.build_solver(model)
