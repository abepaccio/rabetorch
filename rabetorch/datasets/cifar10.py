import torchvision
import torchvision.transforms as transforms

from rabetorch.datasets.base import BaseDataset

class TorchCIFAR10(BaseDataset):
    def __init__(self, _cfg) -> None:
        super().__init__(_cfg)
        self.is_train = _cfg.IS_TRAIN if hasattr(_cfg, "IS_TRAIN") else False

    def build_dataset(self):
        transform = self.build_transform()
        self.dataset = torchvision.datasets.CIFAR10(
            root=self.data_path, train=self.is_train, download=True, transform=transform
        )
