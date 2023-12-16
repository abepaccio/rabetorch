import torch
import torch.nn as nn

from rabetorch.models.backbone.basic_vgg import BasicVgg
from rabetorch.models.custom_model import CustomModel
from rabetorch.models.head.basic_classifier import BasiClassifier


_backbones = {
    "BasicVGG": BasicVgg,
}
_necks = {}
_heads = {
    "BasiClassifier": BasiClassifier,
}

class ModelBuilder():
    def __init__(self, model_cfg) -> None:
        super(ModelBuilder, self).__init__()
        self._cfg = model_cfg
        self.backbone_type = _backbones[model_cfg.BACKBONE.TYPE]
        self.neck_type = _necks[model_cfg.NECK.TYPE] if hasattr(model_cfg, "NECK") else None
        self.head_type = _heads[model_cfg.HEAD.TYPE]

    def build_model(self) -> nn.Module:
        """Build model module.

        Returns:
            nn.Module: Model modeule.
        """
        backbone = self.backbone_type(self._cfg.BACKBONE)
        neck = self.neck_type(self._cfg.NECK) if self.neck_type else None
        head = self.head_type(self._cfg.HEAD)
        return CustomModel(backbone, neck, head)
