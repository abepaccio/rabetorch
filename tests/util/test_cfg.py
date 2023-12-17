import os
import sys
sys.path.append(os.getcwd())

from rabetorch.util.config import load_config


def test_cfg_load():
    cfg_path = "configs/basic_classifier.yaml"
    load_config(cfg_path, None)
