from .preresnet import preresnet
from .pyramidnet import PyramidNet
from .shakeshake.shake_resnet import ShakeResNet
from .wideresnet import WideResNet


def wrn28_2(num_classes):
    return WideResNet(28, 2, 0, num_classes)


def wrn28_10(num_classes):
    return WideResNet(28, 10, 0, num_classes)


def wrn40_2(num_classes):
    return WideResNet(40, 2, 0, num_classes)


def shakeshake26_2x32d(num_classes):
    return ShakeResNet(26, 32, num_classes)


def shakeshake26_2x96d(num_classes):
    return ShakeResNet(26, 96, num_classes)


def shakeshake26_2x112d(num_classes):
    return ShakeResNet(26, 112, num_classes)


def pyramid(num_classes):
    return PyramidNet("cifar10", depth=272, alpha=200, num_classes=num_classes,
                      bottleneck=True)


def resnet200(num_classes):
    return preresnet.preresnet200b(num_classes=num_classes)
