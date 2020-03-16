import homura
import hydra
import torch
from homura import optim, lr_scheduler, reporters, callbacks, trainers
from torch.nn import functional as F

from data import get_dataloader
from models import get_model


def train_and_eval(cfg):
    train_loader, val_loader, test_loader, num_classes = get_dataloader(cfg.data.name,
                                                                        cfg.data.val_size,
                                                                        cfg.data.batch_size,
                                                                        cfg.data.dowload,
                                                                        cfg.augment,
                                                                        False)
    model = get_model(cfg.model.name, num_classes)
    optimizer = optim.SGD(cfg.optim.model.lr, momentum=0.9, weight_decay=cfg.optim.model.weight_decay)
    scheduler = lr_scheduler.MultiStepLR([100, 150])
    tq = reporters.TQDMReporter(range(cfg.optim.epochs), verb=cfg.verb)
    callback = [callbacks.AccuracyCallback(),
                callbacks.LossCallback(),
                reporters.TensorboardReporter("."),
                reporters.IOReporter("."),
                tq]

    with trainers.SupervisedTrainer(model,
                                    optimizer,
                                    F.cross_entropy,
                                    callbacks=callback,
                                    scheduler=scheduler) as trainer:
        for ep in tq:
            trainer.train(train_loader)
            trainer.test(val_loader, 'val')
            trainer.test(test_loader)


@hydra.main('config/main.yaml')
def main(cfg):
    print(cfg.pretty())
    if torch.cuda.is_available():
        torch.cuda.set_device(cfg.gpu_id)
    with homura.set_seed(cfg.seed):
        return train_and_eval(cfg)


if __name__ == '__main__':
    main()
