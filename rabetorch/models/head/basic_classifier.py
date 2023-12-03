import torch
import torch.nn as nn
import torch.nn.functional as F


class BasiClassifier(nn.Module):
    """Basic Classifier head.
    
    This head simply connected with Linear layers.
    """
    def __init__(self, head_cfg):
        """Initialization of basic classifier head component."""
        super().__init__()
        self.num_layer = head_cfg.NUM_LAYER
        self.in_channel = head_cfg.IN_CHANNEL
        self.out_channels = head_cfg.OUT_CHANNEL

        # check layer config
        assert len(self.out_channels) == self.num_layer

        # prepare convolutions
        self.fcs = []
        for layer_idx in range(self.num_layer):
            if layer_idx == 0:
                _fc = nn.Linear(self.in_channel, self.out_channels[layer_idx])
            else:
                _fc = nn.Linear(self.out_channels[layer_idx - 1], self.out_channels[layer_idx])
            self.fcs.append(_fc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward path of classifier model.

        Args:
            x (torch.Tensor): Input of head.

        Returns:
            torch.Tensor: Output of head.
        """
        for _fc in self.fcs:
            x = F.relu(_fc(x))
        return x
