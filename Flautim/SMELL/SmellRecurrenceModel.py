from flautim.pytorch.Model import Model
from torchvision import models
import torch.nn as nn


class RecurrenceModel(Model):
    def __init__(self, context, emb_dim, num_channels=6, **kwargs):
        super(RecurrenceModel, self)\
            .__init__(context, name="RecurrenceNet", **kwargs)
        
        self.emb_dim = emb_dim

        self.base_model = models.mobilenet_v2(weights='DEFAULT')
        self.base_model.features[0][0] = nn.Conv2d(
            in_channels=num_channels,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )

        # instead of 10-class output, project to emb_dim
        self.base_model.classifier[1] = nn.Sequential(
            nn.Linear(84, self.emb_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.base_model(x)
