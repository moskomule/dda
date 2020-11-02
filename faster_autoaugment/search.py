import pathlib
from dataclasses import dataclass
from typing import Any, Mapping, Tuple

import homura
import hydra
import torch
from homura import optim, reporters, trainers
from homura.vision import DATASET_REGISTRY
from torch import Tensor, nn
from torch.nn import functional as F

from policy import Policy
from utils import Config, MODEL_REGISTRY


class Discriminator(nn.Module):
    def __init__(self,
                 base_module: nn.Module
                 ):
        super(Discriminator, self).__init__()
        self.base_model = base_module
        num_features = self.base_model.fc.in_features
        num_class = self.base_model.fc.out_features
        self.base_model.fc = nn.Identity()
        self.classifier = nn.Linear(num_features, num_class)
        self.discriminator = nn.Sequential(nn.Linear(num_features, num_features),
                                           nn.ReLU(),
                                           nn.Linear(num_features, 1))

    def forward(self,
                input: Tensor
                ) -> Tuple[Tensor, Tensor]:
        x = self.base_model(input)
        return self.classifier(x), self.discriminator(x).view(-1)


class AdvTrainer(trainers.TrainerBase):
    # acknowledge https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py
    def iteration(self,
                  data: Tuple[Tensor, Tensor]) -> None:
        # input: [-1, 1]
        input, target = data
        b = input.size(0) // 2
        a_input, a_target = input[:b], target[:b]
        n_input, n_target = input[b:], target[b:]
        loss, d_loss, a_loss = self.wgan_loss(n_input, n_target, a_input, a_target)
        self.reporter.add('loss', loss.detach())
        self.reporter.add('d_loss', d_loss.detach())
        self.reporter.add('a_loss', a_loss.detach())

    def wgan_loss(self,
                  n_input: Tensor,
                  n_target: Tensor,
                  a_input: Tensor,
                  a_target: Tensor
                  ) -> Tuple[Tensor, Tensor, Tensor]:
        ones = n_input.new_tensor(1.0)
        self.model['main'].requires_grad_(True)
        self.model['main'].zero_grad()
        # real images
        output, n_output = self.model['main'](n_input)
        loss = self.cfg.cls_factor * F.cross_entropy(output, n_target)
        loss.backward(retain_graph=True)
        d_n_loss = n_output.mean()
        d_n_loss.backward(-ones)

        # augmented images
        with torch.no_grad():
            # a_input [-1, 1] -> [0, 1]
            a_input = self.model['policy'].denormalize_(a_input)
            augmented = self.model['policy'](a_input)

        _, a_output = self.model['main'](augmented)
        d_a_loss = a_output.mean()
        d_a_loss.backward(ones)
        gp = self.cfg.gp_factor * self.gradient_penalty(n_input, augmented)
        gp.backward()
        self.optimizer['main'].step()

        # train policy
        self.model['main'].requires_grad_(False)
        self.model['policy'].zero_grad()
        _output, a_output = self.model['main'](self.model['policy'](a_input))
        _loss = self.cfg.cls_factor * F.cross_entropy(_output, a_target)
        _loss.backward(retain_graph=True)
        a_loss = a_output.mean()
        a_loss.backward(-ones)
        self.optimizer['policy'].step()

        return loss + _loss, -d_n_loss + d_a_loss + gp, -a_loss

    def gradient_penalty(self,
                         real: Tensor,
                         fake: Tensor
                         ) -> Tensor:
        alpha = real.new_empty(real.size(0), 1, 1, 1).uniform_(0, 1)
        interpolated = alpha * real + (1 - alpha) * fake
        interpolated.requires_grad_()
        _, output = self.model['main'](interpolated)
        grad = torch.autograd.grad(outputs=output, inputs=interpolated, grad_outputs=torch.ones_like(output),
                                   create_graph=True, retain_graph=True, only_inputs=True)[0]
        return (grad.norm(2, dim=1) - 1).pow(2).mean()

    def state_dict(self
                   ) -> Mapping[str, Any]:
        policy: Policy = self.accessible_model['policy']
        return {'policy': policy.state_dict(),
                'policy_kwargs': dict(num_sub_policies=policy.num_sub_policies,
                                      temperature=policy.temperature,
                                      operation_count=policy.operation_count),
                'epoch': self.epoch,
                'step': self.step}

    def save(self,
             path: str
             ) -> None:
        if homura.is_master():
            path = pathlib.Path(path)
            path.mkdir(exist_ok=True, parents=True)
            with (path / f'{self.epoch}.pt').open('wb') as f:
                torch.save(self.state_dict(), f)


@dataclass
class ModelConfig:
    cls_factor: float
    gp_factor: float
    temperature: float
    num_sub_policies: int
    num_chunks: int
    operation_count: int


@dataclass
class DataConfig:
    name: str
    train_size: int
    batch_size: int

    cutout: bool
    download: bool


@dataclass
class OptimConfig:
    epochs: int

    main_lr: float
    policy_lr: float


@dataclass
class BaseConfig(Config):
    model: ModelConfig
    data: DataConfig
    optim: OptimConfig


def search(cfg: BaseConfig
           ):
    train_loader, _, num_classes = DATASET_REGISTRY(cfg.data.name)(batch_size=cfg.data.batch_size,
                                                                   train_size=cfg.data.train_size,
                                                                   drop_last=True,
                                                                   download=cfg.data.download,
                                                                   return_num_classes=True,
                                                                   num_workers=4)
    model = {'main': Discriminator(MODEL_REGISTRY('wrn40_2')(num_classes)),
             'policy': Policy.faster_auto_augment_policy(cfg.model.num_sub_policies,
                                                         cfg.model.temperature,
                                                         cfg.model.operation_count,
                                                         cfg.model.num_chunks)}
    optimizer = {'main': optim.Adam(lr=cfg.optim.main_lr, betas=(0, 0.999)),
                 'policy': optim.Adam(lr=cfg.optim.policy_lr, betas=(0, 0.999))}
    with AdvTrainer(model,
                    optimizer,
                    F.cross_entropy,
                    reporters=[reporters.TensorboardReporter(".")],
                    cfg=cfg.model,
                    use_cuda_nonblocking=True) as trainer:
        for _ in trainer.epoch_range(cfg.optim.epochs):
            trainer.train(train_loader)
        trainer.save(pathlib.Path(hydra.utils.get_original_cwd()) / 'policy_weights' / cfg.data.name)


@hydra.main('config/search.yaml')
def main(cfg: BaseConfig):
    print(cfg.pretty())
    if torch.cuda.is_available():
        torch.cuda.set_device(cfg.gpu)
    with homura.set_seed(cfg.seed):
        return search(cfg)


if __name__ == '__main__':
    main()
