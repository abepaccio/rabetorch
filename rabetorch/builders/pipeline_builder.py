import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

from rabetorch.datasets.cifar10 import TorchCIFAR10

_datasets = {
    "TorchCIFAR10": TorchCIFAR10
}

class PipelineBuilder():
    def __init__(self, data_cfg, is_train) -> None:
        self.data_root = data_cfg.DATA_ROOT
        self.ds_cfgs = data_cfg.TRAIN_DATA if is_train else data_cfg.TEST_DATA
        self.is_train = is_train
        self.batch_size = data_cfg.TOTAL_BATCH
        self.is_custom_sampler = data_cfg.CUSTOM_SAMPLER if hasattr(data_cfg, "CUSTOM_SAMPLER") else False

    def build_dataset(self):
        ds = []
        for ds_cfg in self.ds_cfgs:
            ds_class = _datasets[ds_cfg.TYPE](ds_cfg, self.is_train)
            _ds = ds_class.build_dataset()
            ds_size = int(len(_ds) * ds_cfg.SAMPLING_RATIO) if self.is_custom_sampler else 1
            ds.append(Subset(_ds, range(ds_size)))
        return ds

    def build_pipeline(self):
        ds_list = self.build_dataset()
        ds = ConcatDataset(ds_list)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=self.is_train)
