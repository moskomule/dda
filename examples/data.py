import shutil
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms
from torchvision.datasets import VisionDataset, CIFAR10, CIFAR100, SVHN, ImageFolder, utils as dataset_utils

from utils import RandAugment


class TinyImageNet(ImageFolder):

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        root = Path(root).expanduser()
        if not root.exists() and download:
            root.mkdir(exist_ok=True, parents=True)
            self.download(root)
        if root.exists() and download:
            print("Files already downloaded")
        root = root / "train" if train else root / "val"
        super(TinyImageNet, self).__init__(root, transform=transform, target_transform=target_transform)
        self.data = [self.loader(img) for img, _ in self.samples]

    @staticmethod
    def download(root: Path):
        dataset_utils.download_and_extract_archive(url="http://cs231n.stanford.edu/tiny-imagenet-200.zip",
                                                   download_root=root)
        temp_dir = "tiny-imagenet-200"
        shutil.move(str(root / temp_dir / "train"), str(root))
        shutil.move(str(root / temp_dir / "val"), str(root))
        val_dir = root / "val"
        with (val_dir / "val_annotations.txt").open() as f:
            val_anns = f.read()
        name_to_cls = {i[0]: i[1] for i in
                       [l.split() for l in val_anns.strip().split("\n")]}
        for name, cls in name_to_cls.items():
            (val_dir / cls).mkdir(exist_ok=True)
            shutil.move(str(val_dir / "images" / name), str(val_dir / cls))
        shutil.rmtree(str(root / temp_dir))

    def __len__(self):
        return len(self.data)


class OriginalSVHN(SVHN):
    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False):
        super(OriginalSVHN, self).__init__(root, split="train" if train else "test", transform=transform,
                                           target_transform=target_transform, download=download)
        self.data = [Image.fromarray(np.transpose(img, (1, 2, 0))) for img in self.data]
        self.targets = self.labels


class ExtraSVHN(object):
    def __new__(cls,
                root,
                train=True,
                transform=None,
                target_transform=None,
                download=False):
        if train:
            return (SVHN(root, split='train', transform=transform, download=download) +
                    SVHN(root, split='extra', transform=transform, download=download))
        else:
            return OriginalSVHN(root, train=False, transform=transform, download=download)


def get_dataloader(name: str,
                   val_size: int,
                   batch_size: int,
                   download: bool,
                   augment: Optional[OmegaConf],
                   skip_normalize: bool) -> (DataLoader, DataLoader, DataLoader, int):
    train_set, val_set, test_set, num_classes = _get_dataset(name, val_size, download, augment, skip_normalize)
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=RandomSampler(train_set, True),
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=2 * batch_size, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=2 * batch_size, pin_memory=True)
    return train_loader, val_loader, test_loader, num_classes


def _get_dataset(name: str,
                 val_size: int,
                 download: bool,
                 augment: Optional[OmegaConf],
                 skip_normalize: bool) -> (VisionDataset, VisionDataset, VisionDataset, int):
    _datasets = {"cifar10": (CIFAR10, "~/.torch/data/cifar10",
                             [transforms.ToTensor(),
                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))],
                             [transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                              transforms.RandomHorizontalFlip()], 10),
                 "cifar100": (CIFAR100, "~/.torch/data/cifar100",
                              [transforms.ToTensor(),
                               transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))],
                              [transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                               transforms.RandomHorizontalFlip()], 100),
                 "svhn": (OriginalSVHN, "~/.torch/data/svhn",
                          [transforms.ToTensor(),
                           transforms.Normalize((0.4390, 0.4443, 0.4692), (0.1189, 0.1222, 0.1049))],
                          [transforms.RandomCrop(32, padding=4, padding_mode='reflect')], 10),
                 "extrasvhn": (ExtraSVHN, "~/.torch/data/svhn",
                               [transforms.ToTensor(),
                                transforms.Normalize((0.4390, 0.4443, 0.4692), (0.1189, 0.1222, 0.1049))],
                               [transforms.RandomCrop(32, padding=4, padding_mode='reflect')], 10),
                 "tinyimagenet": (TinyImageNet, "~/.torch/data/tinyimagenet",
                                  [transforms.ToTensor(),
                                   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))],
                                  [transforms.Resize(40),
                                   transforms.RandomCrop(32),
                                   transforms.RandomHorizontalFlip()], 200)
                 }

    if name not in _datasets.keys():
        raise RuntimeError(f'Unknown dataset name {name}')

    dset, root, norm_transform, data_aug, num_cls = _datasets[name]
    train_set = dset(root, train=True, transform=transforms.Compose(norm_transform), download=download)
    test_set = dset(root, train=False, transform=transforms.Compose(norm_transform), download=download)
    train_set, val_set = _split_dataset(train_set, val_size)

    if augment is not None and hasattr(augment, 'name'):
        if str(augment.name).lower() == 'randaugment':
            print('Apply randaugment')
            data_aug += [RandAugment(augment.num_aug, augment.magnitude)]
        else:
            raise NotImplementedError

    if skip_normalize:
        norm_transform = norm_transform[:1]
    train_set.transform = transforms.Compose(data_aug + norm_transform)

    return train_set, val_set, test_set, num_cls


def _split_dataset(dataset: VisionDataset,
                   val_size: int) -> (VisionDataset, VisionDataset):
    indices = torch.randperm(len(dataset))
    valset = deepcopy(dataset)
    dataset.data = [dataset.data[i] for i in indices[val_size:]]
    dataset.targets = [dataset.targets[i] for i in indices[val_size:]]

    valset.data = [valset.data[i] for i in indices[:val_size]]
    valset.targets = [valset.targets[i] for i in indices[:val_size]]

    return dataset, valset
