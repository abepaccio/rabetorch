import os
import sys
from omegaconf import OmegaConf
sys.path.append(os.getcwd())

from rabetorch.builders.pipeline_builder import PipelineBuilder

def test_build_pipeline():
    # set config
    cfg_dict = {
        "DATA_ROOT": "",
        "TRAIN_DATA": [
            {
                "TYPE": "TorchCIFAR10",
                "TRANSFORM": [{"TYPE": "ToTensor"}],
                "DATA_PATH": "./data",
            },
        ],
        "TOTAL_BATCH": 1,
    }
    cfg = OmegaConf.create(cfg_dict)

    pipeline_builder = PipelineBuilder(cfg, is_train=True)
    pipeline_builder.build_pipeline()
