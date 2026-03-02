from flautim.pytorch.Model import Model
from torchvision import models
import torch.nn as nn
import flautim as fl
import torch.nn.functional as F
import torch


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
            
    def forward_embedding(self, x):
        """
        Returns the representation BEFORE the final classifier logits.
        For torchvision MobileNetV2 this is pooled features of dim = last_channel (1280).
        """
        x = self.base_model.features(x)                      # [B, 1280, H', W']
        x = F.adaptive_avg_pool2d(x, (1, 1))                 # [B, 1280, 1, 1]
        x = torch.flatten(x, 1)                              # [B, 1280]
        return x
    def forward(self, x):
        try:
            return self.base_model(x)
        except Exception as e:
            fl.log(f"Error in RecurrenceModel __forward__(): {e}")
