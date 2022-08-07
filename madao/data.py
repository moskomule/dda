from homura.vision import DATASET_REGISTRY
import copy
import dataclasses

import torch
from torch.utils.data import DataLoader
from typing import Any
from torchvision.transforms import ToTensor, Normalize, RandomErasing


class Cycle(object):
    def __init__(self, data_loader: DataLoader, length: int = -1):
        self.data_loader = data_loader
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        counter = 0
        while True:
            for data in self.data_loader:
                if counter == self.length:
                    return
                yield data
                counter += 1


@dataclasses.dataclass
class Data:
    train_da: torch.Tensor
    train_no_da: torch.Tensor
    train_labels_da: torch.Tensor
    train_labels_no_da: torch.Tensor
    val: torch.Tensor
    val_labels: torch.Tensor

    def to(self, *args, **kwargs):
        return Data(*[d.to(*args, **kwargs) if isinstance(d, torch.Tensor) else d
                      for d in dataclasses.astuple(self)])


@dataclasses.dataclass
class TrainLoader:
    train_da_loader: DataLoader
    train_no_da_loader: DataLoader
    val_loader: DataLoader
    train_da: Any
    da_interval: int
    mean_std: tuple[tuple, tuple]

    def __post_init__(self):
        self.counter = 0
        self.separate_val = self.val_loader is not None
        print('separate validation loader is used!')

    def __len__(self):
        return len(self.train_da_loader)

    def __iter__(self):
        step = 0
        self.counter += 1

        while step < len(self.train_da_loader):
            # shuffle data

            # this is intended to reset DataLoaders here so that learned CPU policies are distributed to all workers
            # after each self.da_interval step

            train_da_loader = iter(self.train_da_loader)
            train_no_da_loader = iter(self.train_no_da_loader)
            da_interval = self.da_interval if self.da_interval > 0 else len(train_no_da_loader)

            for packed in zip(range(da_interval), train_da_loader, train_no_da_loader, Cycle(self.val_loader)):
                step += 1
                train_da, train_labels_da = packed[1]
                train_no_da, train_labels_no_da = packed[2]
                val, val_labels = packed[3]
                yield Data(train_da, train_no_da, train_labels_da, train_labels_no_da, val, val_labels)

                if step == len(self.train_da_loader):
                    del train_da_loader
                    del train_no_da_loader
                    return

    def register_policy(self, policy):
        tnl = self.train_no_da_loader
        dataset = copy.deepcopy(tnl.dataset)
        dataset.transform.transforms = self.train_da + [policy.pil_forward] + [ToTensor(),
                                                                               Normalize(*self.mean_std),
                                                                               RandomErasing()]
        train_loader = DataLoader(dataset, batch_size=tnl.batch_size, shuffle=True, num_workers=tnl.num_workers,
                                  pin_memory=tnl.pin_memory)
        self.train_da_loader = train_loader


def get_data(cfg):
    ds = DATASET_REGISTRY(cfg.name).setup(batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                                          pin_memory=cfg.pin_memory, download=cfg.download, train_size=cfg.train_size,
                                          val_size=cfg.val_size, post_norm_train_da=[RandomErasing()])
    norm = ds.default_norm[-1]
    train_loader = ds.train_loader
    train_loader.dataset.transform.transforms.pop(-2)
    return TrainLoader(None, train_loader, ds.val_loader, ds.default_train_da, cfg.da_interval,
                       (norm.mean, norm.std)), ds.test_loader, ds.num_classes
