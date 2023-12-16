import torch
import torch.nn as nn
import torch.optim as optim


class CustomBackbone(nn.Module):
    def __init__(self):
        super(CustomBackbone, self).__init__()

    def forward(self, x):
        raise NotImplementedError


class CustomNeck(nn.Module):
    def __init__(self):
        super(CustomNeck, self).__init__()

    def forward(self, x):
        raise NotImplementedError



class CustomHead(nn.Module):
    def __init__(self):
        super(CustomHead, self).__init__()

    def forward(self, x):
        raise NotImplementedError


class CustomModel(nn.Module):
    def __init__(self, backbone, neck, head):
        super(CustomModel, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, x):
        x = self.backbone(x)
        if self.neck:
            x = self.neck(x)
        x = self.head(x)
        return x
