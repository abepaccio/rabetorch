import torchvision
import torchvision.transforms as transforms

from rabetorch.datasets.base import BaseDataset

class TorchCIFAR10(BaseDataset):
    def __init__(self, _cfg, is_train:bool = True) -> None:
        super(TorchCIFAR10, self).__init__(_cfg)
        self.is_train = is_train

    def build_dataset(self):
        transform = self.build_transform()
        self.dataset = torchvision.datasets.CIFAR10(
            root=self.data_path,
            train=self.is_train,
            download=True,
            transform=transform
        )
        return self.dataset

    def __len__(self):
        return len(self.dataset)
