from rabetorch.datasets.cifar10 import TorchCIFAR10

_datasets = {
    "TorchCIFAR10": TorchCIFAR10
}

class DatasetBuilder():
    def __init__(self, data_cfg) -> None:
        self.data_loader = data_cfg.DATA_LOADER
        self.data_root = data_cfg.DATA_ROOT
        self.train_ds_cfg = data_cfg.TRAIN_DATA
        self.test_ds_cfg = data_cfg.TEST_DATA

    def build_dataset(self, is_tain):
        ds_cfgs = self.train_ds_cfg if is_tain else self.test_ds_cfg
        ds = []
        for ds_cfg in ds_cfgs:
            dataset = _datasets[ds_cfg.TYPE](ds_cfg)
            dataset.build_dataset()
            ds.append(dataset)
        return ds
