from rabetorch.datasets.data_loader import BasicDataLoader

_data_loader = {
    "BasicDataLoader": BasicDataLoader
}

class PipelineBuilder():
    def __init__(self, _data_cfg, is_train: bool):
        self._data_cfg = _data_cfg
        self.is_train = is_train
        self.loader_type = _data_loader[_data_cfg.DATA_LOADER]

    def build_pipeline(self, datasets):
        loader = self.loader_type(self._data_cfg, self.is_train)
        loader.build_dataloader(datasets)
