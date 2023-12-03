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

    def build_train_dataset(self):
        train_datasets = []
        for ds_cfg in self.train_ds_cfg:
            dataset = _datasets[ds_cfg.TYPE](ds_cfg)
            dataset.build_dataset()
            train_datasets.append(dataset)

    def build_test_dataset(self):
        test_datasets = []
        for ds_cfg in self.test_ds_cfg:
            dataset = _datasets[ds_cfg.TYPE](ds_cfg)
            dataset.build_dataset()
            test_datasets.append(dataset)

# あとはloader, optimeizer, modelのコンパイル, ロスの登録, 学習パイプラインの記述, 評価パイプラインの記述
