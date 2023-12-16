import os
import sys
sys.path.append(os.getcwd())

import torchvision
import torchvision.transforms as transforms

from rabetorch.builders.dataset_builder import DatasetBuilder
from rabetorch.builders.pipeline_builder import PipelineBuilder
from rabetorch.datasets.data_loader import BasicDataLoader
from rabetorch.util.config import parse_dict_config, _print_attributes
from rabetorch.util.io_util import load_yaml


def test_build_pipeline_with_basic_dataloader():
    # load config
    yaml_fp = "configs/basic_classifier.yaml"
    base_cfg_dict = load_yaml(yaml_fp)
    cfg = parse_dict_config(base_cfg_dict)
    if hasattr(cfg, "BASE"):
        for sub_cfg_path in base_cfg_dict.get("BASE", None):
            sub_cfg_dict = load_yaml("configs/" + sub_cfg_path)
            _cfg = parse_dict_config(sub_cfg_dict)
            cfg.update(_cfg)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # build pipeline
    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    pipeline = PipelineBuilder(cfg.DATA, is_train=True)
    pipeline.build_pipeline(train_set)
