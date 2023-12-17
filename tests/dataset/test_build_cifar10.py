import os
import sys
from omegaconf import OmegaConf
sys.path.append(os.getcwd())

from rabetorch.datasets.cifar10 import TorchCIFAR10


def test_build_cifar10():
    # set config
    cfg_dict = {
        "TRANSFORM": [{"TYPE": "ToTensor"}],
        "DATA_PATH": "./data",
    }
    cfg = OmegaConf.create(cfg_dict)

    # build dataset
    torch_cifar_10 = TorchCIFAR10(cfg, is_train=True)
    torch_cifar_10.build_dataset()
    len(torch_cifar_10)
