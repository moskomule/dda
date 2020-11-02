from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Tuple

import homura
import hydra
import torch
from homura import lr_scheduler, optim, reporters, trainers
from homura.metrics import accuracy
from homura.vision import DATASET_REGISTRY
from torch import Tensor
from torch.nn import functional  as F
from torchvision import transforms

from policy import Policy
from utils import Config, MODEL_REGISTRY


class EvalTrainer(trainers.TrainerBase):
    def __init__(self, *args, **kwargs):
        super(EvalTrainer, self).__init__(*args, **kwargs)
        if self.policy is not None:
            self.policy.to(self.device)
            self.policy.eval()

    def iteration(self,
                  data: Tuple[Tensor, Tensor]
                  ) -> Mapping[str, Tensor]:
        # input [-1, 1]
        input, target = data
        if self.policy is not None and self.is_train:
            with torch.no_grad():
                # input: [-1, 1]
                input = self.policy(self.policy.denormalize_(input))
        output = self.model(input)
        loss = self.loss_f(output, target)
        if self.is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.reporter.add('loss', loss.detach())
        self.reporter.add('accuracy', accuracy(output, target))


@dataclass
class ModelConfig:
    name: str
    num_chunks: int


@dataclass
class DataConfig:
    name: str
    batch_size: int
    download: bool


@dataclass
class CosineSchedulerConfig:
    mul: float
    warmup: int


@dataclass
class StepSchedulerConfig:
    steps: List[int]
    gamma: float


@dataclass
class OptimConfig:
    epochs: int
    lr: float
    momentum: float
    weight_decay: float
    nesterov: bool

    scheduler: CosineSchedulerConfig or StepSchedulerConfig


@dataclass
class BaseConfig(Config):
    path: str

    model: ModelConfig
    data: DataConfig
    optim: OptimConfig


def train_and_eval(cfg: BaseConfig):
    if cfg.path is None:
        print('cfg.path is None, so FasterAutoAugment is not used')
        policy = None
    else:
        path = Path(hydra.utils.get_original_cwd()) / cfg.path
        assert path.exists()
        policy_weight = torch.load(path, map_location='cpu')
        policy = Policy.faster_auto_augment_policy(num_chunks=cfg.model.num_chunks, **policy_weight['policy_kwargs'])
        policy.load_state_dict(policy_weight['policy'])
    train_loader, test_loader, num_classes = DATASET_REGISTRY(cfg.data.name)(batch_size=cfg.data.batch_size,
                                                                             drop_last=True,
                                                                             download=cfg.data.download,
                                                                             return_num_classes=True,
                                                                             norm=[transforms.ToTensor(),
                                                                                   transforms.Normalize(
                                                                                       (0.5, 0.5, 0.5),
                                                                                       (0.5, 0.5, 0.5))
                                                                                   ],
                                                                             num_workers=4
                                                                             )
    model = MODEL_REGISTRY(cfg.model.name)(num_classes)
    optimizer = optim.SGD(cfg.optim.lr,
                          momentum=cfg.optim.momentum,
                          weight_decay=cfg.optim.weight_decay,
                          nesterov=cfg.optim.nesterov)
    scheduler = lr_scheduler.CosineAnnealingWithWarmup(cfg.optim.epochs,
                                                       cfg.optim.scheduler.mul,
                                                       cfg.optim.scheduler.warmup)

    with EvalTrainer(model,
                     optimizer,
                     F.cross_entropy,
                     reporters=[reporters.TensorboardReporter(".")],
                     scheduler=scheduler,
                     policy=policy,
                     cfg=cfg.model,
                     use_cuda_nonblocking=True) as trainer:
        for _ in trainer.epoch_range(cfg.optim.num_epochs):
            trainer.train(train_loader)
            trainer.test(test_loader)
    print(f"Min. Error Rate: {1 - max(c[1].history['test']):.3f}")


@hydra.main('config/train.yaml')
def main(cfg: BaseConfig):
    print(cfg.pretty())
    if torch.cuda.is_available():
        torch.cuda.set_device(cfg.gpu)
    with homura.set_seed(cfg.seed):
        return train_and_eval(cfg)


if __name__ == '__main__':
    main()
