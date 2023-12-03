import os
import sys
sys.path.append(os.getcwd())

from rabetorch.builders.dataset_builder import DatasetBuilder
from rabetorch.util.config import parse_dict_config, _print_attributes
from rabetorch.util.io_util import load_yaml


def test_build_cifar10():
    # load config
    yaml_fp = "configs/basic_classifier.yaml"
    base_cfg_dict = load_yaml(yaml_fp)
    cfg = parse_dict_config(base_cfg_dict)
    if hasattr(cfg, "BASE"):
        for sub_cfg_path in base_cfg_dict.get("BASE", None):
            sub_cfg_dict = load_yaml("configs/" + sub_cfg_path)
            _cfg = parse_dict_config(sub_cfg_dict)
            cfg.update(_cfg)

    # build model
    ds_builder = DatasetBuilder(cfg.DATA)
    ds_builder.build_train_dataset()
    ds_builder.build_test_dataset()
