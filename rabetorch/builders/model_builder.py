import torch
import torch.nn as nn

from rabetorch.models.backbone.basic_vgg import BasicVgg
from rabetorch.models.head.basic_classifier import BasiClassifier

_backbones = {
    "BasicVgg": BasicVgg,
}
_necks = {}
_heads = {
    "BasiClassifier": BasiClassifier,
}

class ModelBuilder():
    def __init__(self, model_cfg) -> None:
        self._cfg = model_cfg
        self.backbone_type = model_cfg.BACKBONE.TYPE
        self.head_type = model_cfg.HEAD.TYPE
        self.neck_type = model_cfg.NECK.TYPE if hasattr(model_cfg, "NECK") else None
        self.backbone_module = _backbones[self.backbone_type]
        self.head_module = _heads[self.head_type]
        self.neck_module = _necks[self.neck_type] if self.neck_type else None

    def build_model(self) -> nn.Module:
        """Build model module.

        Returns:
            nn.Module: Model modeule.
        """
        self.backbone = self.backbone_module(self._cfg.BACKBONE)
        if self.neck_module:
            self.neck = self.neck_module(self._cfg.NECK)
        self.head = self.head_module(self._cfg.HEAD)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward path of model

        Args:
            x (torch.Tensor):Input of model.

        Returns:
            torch.Tensor: Output of model.
        """
        x = self.backbone.forward(x)
        if self.neck_module:
            x = self.neck.forward(x)
        return self.head.forward(x)
