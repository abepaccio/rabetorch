import torch

from rabetorch.datasets.base import MultiDataset


class BasicDataLoader():
    def __init__(self, _dataset_cfg, do_shuffle: bool):
        self._dataset_cfg = _dataset_cfg
        self.total_batch = _dataset_cfg.TOTAL_BATCH
        self.num_gpu = _dataset_cfg.NUM_GPU
        self.do_shuffle = do_shuffle

    def build_dataloader(self, datasets):
        self.multi_dataset = MultiDataset(datasets)
        data_loader = torch.utils.data.DataLoader(
            self.multi_dataset,
            batch_size=self.total_batch,
            shuffle=self.do_shuffle,
            num_workers=self.num_gpu
        )
        return data_loader
