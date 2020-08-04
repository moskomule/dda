from dataclasses import dataclass

from homura.vision import MODEL_REGISTRY
from homura.vision.models.classification.wideresnet import WideResNet


@dataclass
class Config:
    gpu: int
    seed: int

    def pretty(self):
        pass


@MODEL_REGISTRY.register
def wrn40_2(num_classes=10, dropout_rate=0) -> WideResNet:
    model = WideResNet(depth=40, widen_factor=2, dropout_rate=dropout_rate, num_classes=num_classes)
    return model
