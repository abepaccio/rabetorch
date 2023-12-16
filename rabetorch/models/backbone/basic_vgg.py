import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicVgg(nn.Module):
    """Basic Vgg Backbone.
    
    This backbone simply accumulate convolution layer with activations.
    """
    def __init__(self, backbone_cfg):
        """Initialization of basic vgg backbone component."""
        super().__init__()
        self.num_layer = backbone_cfg.NUM_LAYER
        self.out_channels = backbone_cfg.OUT_CHANNEL
        self.kernel_sizes = backbone_cfg.KERNEL_SIZE
        self.strides = backbone_cfg.STRIDE if hasattr(backbone_cfg, "STRIDE") else [1 for _ in range(self.num_layer)]
        self.paddings = backbone_cfg.PADDING if hasattr(backbone_cfg, "PADDING") else [1 for _ in range(self.num_layer)]
        self.maxxpool_strides = backbone_cfg.MAXPOOL_STRIDE if hasattr(backbone_cfg, "MAXPOOL_STRIDE") else [False for _ in range(self.num_layer)]
        self.flatten_out = backbone_cfg.FLATTEN_OUT if hasattr(backbone_cfg, "FLATTEN_OUT") else False

        # check layer config
        assert len(self.strides) == self.num_layer
        assert len(self.paddings) == self.num_layer
        assert len(self.maxxpool_strides) == self.num_layer
        assert len(self.out_channels) == self.num_layer
        assert len(self.kernel_sizes) == self.num_layer

        # prepare convolutions
        self.convs = nn.ModuleList()
        for layer_idx in range(self.num_layer):
            if layer_idx == 0:
                _conv = nn.Conv2d(
                    3, 
                    self.out_channels[layer_idx],
                    self.kernel_sizes[layer_idx],
                    self.strides[layer_idx],
                    self.paddings[layer_idx],
                )
            else:
                _conv = nn.Conv2d(
                    self.out_channels[layer_idx - 1],
                    self.out_channels[layer_idx],
                    self.kernel_sizes[layer_idx],
                    self.strides[layer_idx],
                    self.paddings[layer_idx],
                )
            self.convs.append((_conv))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward path of Vgg model.

        Because this is Vgg, each block considered by convolution wise.

        Args:
            x (torch.Tensor): Input of backbone.

        Returns:
            torch.Tensor: Output of backbone.
        """
        for _conv, maxxpool_stride in zip(self.convs, self.maxxpool_strides):
            x = F.relu(_conv(x))
            if maxxpool_stride:
                x = F.max_pool2d(x, maxxpool_stride)
        if self.flatten_out:
            # output row vector
            return x.view(x.size(0), -1)
        return x
