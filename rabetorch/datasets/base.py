import torchvision
import torchvision.transforms as transforms


_transforms = {
    "ToTensor": transforms.ToTensor,
    "Normalize": transforms.Normalize
}


class BaseDataset():
    def __init__(self, dataset_cfg) -> None:
        self.dataset_cfg = dataset_cfg
        self.transform_cfg = dataset_cfg.TRANSFORM
        self.data_path = dataset_cfg.DATA_PATH
        self.do_shuffle = dataset_cfg.DO_SHUFFLE if hasattr(dataset_cfg, "DO_SHUFFLE") else False

    def build_transform(self) -> transforms:
        transform_list = []
        for _transform in self.transform_cfg:
            transform = _transforms[_transform.TYPE]
            if _transform.TYPE == "ToTensor":
                transform_list.append((transform()))
            elif _transform.TYPE == "Normalize":
                mean = tuple(_transform.MEAN)
                std = tuple(_transform.STD)
                transform_list.append(transform(mean, std))
            else:
                raise TypeError
        return transforms.Compose(transform_list)

    def build_dataset(self):
        pass
