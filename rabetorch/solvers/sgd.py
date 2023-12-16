import torch
import torch.nn as nn


class SGD():
    def __init__(self, _solver_cfg):
        self.base_lr = _solver_cfg.BASE_LR

    def get_solver(self, model):
        params = nn.ParameterList(model.parameters())
        return torch.optim.SGD(params, lr=self.base_lr)
