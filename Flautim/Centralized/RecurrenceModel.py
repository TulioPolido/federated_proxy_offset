from flautim.pytorch.Model import Model
from torchvision import models
import torch.nn as nn
import flautim as fl


class RecurrenceModel(Model):
    def __init__(self, context, num_classes=7, num_channels=6, **kwargs):
        try:
            super(RecurrenceModel, self).__init__(
                context, name="RecurrenceNet", **kwargs
            )

            self.base_model = models.mobilenet_v2(weights="DEFAULT")
            self.base_model.features[0][0] = nn.Conv2d(
                in_channels=num_channels,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            )
            self.base_model.classifier[1] = nn.Linear(
                self.base_model.last_channel, num_classes
            )
        except Exception as e:
            fl.log(f"Error in RecurrenceModel __init__(): {e}")

    def forward(self, x):
        try:
            return self.base_model(x)
        except Exception as e:
            fl.log(f"Error in RecurrenceModel __forward__(): {e}")
