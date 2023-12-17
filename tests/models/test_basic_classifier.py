import os
import sys
from omegaconf import OmegaConf
sys.path.append(os.getcwd())

from rabetorch.builders.model_builder import ModelBuilder


def test_build_model():
    # set config
    cfg_dict = {
        "BACKBONE": {
            "TYPE": "BasicVGG",
            "NUM_LAYER": 1,
            "OUT_CHANNEL": [64],
            "KERNEL_SIZE": [3],
            "FLATTEN_OUT": True,
        },
        "HEAD": {
            "TYPE": "BasiClassifier",
            "NUM_LAYER": 1,
            "IN_CHANNEL": 64,
            "OUT_CHANNEL": [10],
        }
    }
    cfg = OmegaConf.create(cfg_dict)

    # build model
    model_builder = ModelBuilder(cfg)
    model_builder.build_model()
