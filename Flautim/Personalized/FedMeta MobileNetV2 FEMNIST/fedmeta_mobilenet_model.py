from flautim.pytorch.Model import Model
from torchvision import models
import torch.nn as nn


def replace_bn_with_gn(module, num_groups=8):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features

            groups = min(num_groups, num_channels)
            while num_channels % groups != 0:
                groups -= 1

            setattr(module, name, nn.GroupNorm(groups, num_channels))
        else:
            replace_bn_with_gn(child, num_groups)


class MobileNetV2WithoutBatchNorm(Model):
    def __init__(self, context, num_classes=7, num_channels=6, **kwargs):
        super(MobileNetV2WithoutBatchNorm, self).__init__(context, name="RecurrenceNet", **kwargs)

        self.base_model = models.mobilenet_v2(weights="DEFAULT")

        # Change input channels
        self.base_model.features[0][0] = nn.Conv2d(
            in_channels=num_channels,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )

        # Replace classifier
        self.base_model.classifier[1] = nn.Linear(
            self.base_model.last_channel,
            num_classes
        )

        # 🔹 Replace BatchNorm everywhere
        replace_bn_with_gn(self.base_model)

    def forward(self, x):
        return self.base_model(x)