import os
import sys
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as F

from rabetorch.builders.solver_builder import SolverBuilder
from rabetorch.util.config import parse_dict_config
from rabetorch.util.io_util import load_yaml


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
    # load config
    yaml_fp = "configs/basic_classifier.yaml"
    base_cfg_dict = load_yaml(yaml_fp)
    cfg = parse_dict_config(base_cfg_dict)
    if hasattr(cfg, "BASE"):
        for sub_cfg_path in base_cfg_dict.get("BASE", None):
            sub_cfg_dict = load_yaml("configs/" + sub_cfg_path)
            _cfg = parse_dict_config(sub_cfg_dict)
            cfg.update(_cfg)

    # build solver
    model = Net()
    solver = SolverBuilder(cfg.SOLVER)
    solver.build_solver(model)
