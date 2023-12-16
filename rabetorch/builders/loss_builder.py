import torch
import torch.nn as nn

_losses = {
    "CrossEntropyLoss": nn.CrossEntropyLoss
}

class LossBuilder():
    """
    docstring
    """
    def __init__(self, _solver_cfg) -> None:
        self._solver_cfg = _solver_cfg
        self.loss_cfg = _solver_cfg.LOSS
        self.loss_type = self.loss_cfg.TYPE
        self.loss_args = self.loss_cfg.ARGS if hasattr(self.loss_cfg, "ARGS") else None

    def build_loss(self):
        return _losses[self.loss_type](self.loss_args)
