"""fedrep: A Flower Baseline."""

from typing import Tuple

import torch
from torch import nn

from base_model import ModelManager, ModelSplit

class CNNCifar10(nn.Module):
    """CNN model adapted for 128x128x6 input and 7 output classes."""

    def __init__(self):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(6, 64, 5, padding=2),   # (128x128) -> (128x128)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),               # (128x128) -> (64x64)

            nn.Conv2d(64, 128, 5, padding=2), # (64x64) -> (64x64)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),               # (64x64) -> (32x32)

            nn.Conv2d(128, 256, 3, padding=1), # (32x32) -> (32x32)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                # (32x32) -> (16x16)

            nn.Flatten(),
            nn.Linear(256 * 16 * 16, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(128, 7)  # 7 output classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.body(x)
        return self.head(x)


class CNNCifar10ModelSplit(ModelSplit):
    """Split CNNCifar10 model into body and head."""

    def _get_model_parts(self, model: CNNCifar10) -> Tuple[nn.Module, nn.Module]:
        return model.body, model.head


class CNNCifar10ModelManager(ModelManager):
    """Manager for CNNCifar10 models with Body/Head split."""

    def __init__(self, **kwargs):
        super().__init__(model_split_class=CNNCifar10ModelSplit, **kwargs)

    def _create_model(self) -> nn.Module:
        return CNNCifar10()


# pylint: disable=W0223
class CNNCifar100(nn.Module):
    """CNN model for CIFAR100 dataset.

    Refer to
    https://github.com/rahulv0205/fedrep_experiments/blob/main/models/Nets.py
    """

    def __init__(self):
        """Initialize the model."""
        super().__init__()

        # Note that in the official implementation, the body has no BN layers.
        # However, no BN will definitely lead training to collapse.
        self.body = nn.Sequential(
            nn.Conv2d(3, 64, 5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 5),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(128 * 5 * 5, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.6),
        )

        self.head = nn.Sequential(nn.Linear(128, 100))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        x = self.body(x)
        return self.head(x)


class CNNCifar100ModelSplit(ModelSplit):
    """Split CNNCifar100 model into body and head."""

    def _get_model_parts(self, model: CNNCifar100) -> Tuple[nn.Module, nn.Module]:
        return model.body, model.head


# pylint: disable=R0902, R0913, R0801
class CNNCifar100ModelManager(ModelManager):
    """Manager for models with Body/Head split."""

    def __init__(self, **kwargs):
        """Initialize the attributes of the model manager."""
        super().__init__(model_split_class=CNNCifar100ModelSplit, **kwargs)

    def _create_model(self) -> CNNCifar100:
        """Return CNNCifar100 model to be split into head and body."""
        return CNNCifar100()
