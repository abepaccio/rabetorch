import os
import sys
sys.path.append(os.getcwd())

from rabetorch.util.config import parse_dict_config
from rabetorch.util.io_util import load_yaml


def test_cfg_load():
    yaml_fp = "configs/basic_classifier.yaml"
    base_cfg_dict = load_yaml(yaml_fp)
    cfg = parse_dict_config(base_cfg_dict)
    if hasattr(cfg, "BASE"):
        for sub_cfg_path in base_cfg_dict.get("BASE", None):
            sub_cfg_dict = load_yaml("configs/" + sub_cfg_path)
            _cfg = parse_dict_config(sub_cfg_dict)
            cfg.update(_cfg)
