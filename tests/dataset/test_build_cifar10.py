import os
import sys
sys.path.append(os.getcwd())

from rabetorch.builders.dataset_builder import DatasetBuilder
from rabetorch.util.config import load_config


def test_build_cifar10():
    # load config
    yaml_fp = "configs/basic_classifier.yaml"
    cfg = load_config(yaml_fp)

    # build dataset
    ds_builder = DatasetBuilder(cfg.DATA)
    train_data = ds_builder.build_dataset(is_tain=True)
    test_data = ds_builder.build_dataset(is_tain=False)
