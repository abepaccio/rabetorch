import torch


class SGD():
    def __init__(self, _solver_cfg):
        self.base_lr = _solver_cfg.BASE_LR

    def get_solver(self, model):
        return torch.optim.SGD(model.parameters(), lr=self.base_lr)
