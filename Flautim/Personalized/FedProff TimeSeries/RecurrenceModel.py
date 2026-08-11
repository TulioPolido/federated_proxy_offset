from flautim.pytorch.Model import Model

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1D Inverted Residual Block
# ============================================================

class InvertedResidual1D(nn.Module):

    def __init__(
        self,
        inp,
        oup,
        stride,
        expand_ratio,
        norm_layer=nn.BatchNorm1d
    ):
        super().__init__()

        self.stride = stride

        hidden_dim = int(round(inp * expand_ratio))

        self.use_res_connect = (
            stride == 1 and inp == oup
        )

        layers = []

        # ----------------------------------------------------
        # Expansion
        # ----------------------------------------------------

        if expand_ratio != 1:

            layers.extend([
                nn.Conv1d(
                    inp,
                    hidden_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False
                ),

                norm_layer(hidden_dim),

                nn.ReLU6(inplace=True),
            ])

        # ----------------------------------------------------
        # Depthwise convolution
        # ----------------------------------------------------

        layers.extend([
            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=hidden_dim,
                bias=False
            ),

            norm_layer(hidden_dim),

            nn.ReLU6(inplace=True),
        ])

        # ----------------------------------------------------
        # Projection
        # ----------------------------------------------------

        layers.extend([
            nn.Conv1d(
                hidden_dim,
                oup,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),

            norm_layer(oup),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):

        if self.use_res_connect:
            return x + self.conv(x)

        return self.conv(x)


# ============================================================
# 1D MobileNetV2
# ============================================================

class RecurrenceModel(Model):

    def __init__(
        self,
        context,
        num_classes=7,
        num_channels=6,
        batch_norm=True,
        width_mult=1.0,
        **kwargs
    ):

        super(RecurrenceModel, self).__init__(
            context,
            name="RecurrenceModel",
            **kwargs
        )

        # ====================================================
        # Normalization
        # ====================================================

        if batch_norm:

            norm_layer = nn.BatchNorm1d

        else:

            def norm_layer(num_channels):
                groups = min(8, num_channels)

                while num_channels % groups != 0:
                    groups -= 1

                return nn.GroupNorm(
                    groups,
                    num_channels
                )

        # ====================================================
        # MobileNetV2 configuration
        #
        # t = expansion factor
        # c = output channels
        # n = number of repetitions
        # s = stride of first block
        # ====================================================

        inverted_residual_setting = [

            # t,  c,   n,  s

            [1,  16,  1,  1],

            [6,  24,  2,  2],

            [6,  32,  3,  2],

            [6,  64,  4,  2],

            [6,  96,  3,  1],

            [6, 160,  3,  2],

            [6, 320,  1,  1],
        ]

        # ====================================================
        # Channel configuration
        # ====================================================

        def make_divisible(
            value,
            divisor=8
        ):

            return int(
                (value + divisor / 2) // divisor * divisor
            )

        input_channel = make_divisible(
            32 * width_mult
        )

        last_channel = make_divisible(
            1280 * max(1.0, width_mult)
        )

        # ====================================================
        # Feature extractor
        # ====================================================

        features = []

        # ----------------------------------------------------
        # Initial convolution
        #
        # [B, 6, 1024]
        #
        # -> [B, 32, 512]
        # ----------------------------------------------------

        features.append(
            nn.Conv1d(
                num_channels,
                input_channel,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            )
        )

        features.append(
            norm_layer(input_channel)
        )

        features.append(
            nn.ReLU6(inplace=True)
        )

        # ====================================================
        # Inverted residual blocks
        # ====================================================

        for t, c, n, s in inverted_residual_setting:

            output_channel = make_divisible(
                c * width_mult
            )

            for i in range(n):

                stride = (
                    s
                    if i == 0
                    else 1
                )

                features.append(
                    InvertedResidual1D(
                        inp=input_channel,
                        oup=output_channel,
                        stride=stride,
                        expand_ratio=t,
                        norm_layer=norm_layer
                    )
                )

                input_channel = output_channel

        # ====================================================
        # Final 1x1 convolution
        #
        # -> 1280 channels
        # ====================================================

        features.append(
            nn.Conv1d(
                input_channel,
                last_channel,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )
        )

        features.append(
            norm_layer(last_channel)
        )

        features.append(
            nn.ReLU6(inplace=True)
        )

        self.features = nn.Sequential(*features)

        # ====================================================
        # Embedding dimension
        # ====================================================

        self.last_channel = last_channel

        # ====================================================
        # Classifier
        # ====================================================

        self.classifier = nn.Linear(
            last_channel,
            num_classes
        )

    # ========================================================
    # Prepare input
    # ========================================================

    def _prepare_input(self, x):

        """
        Accepts:

            [B, 1024, 6]

        or:

            [B, 6, 1024]

        Internally converts to:

            [B, 6, 1024]
        """

        if x.ndim != 3:

            raise ValueError(
                f"Expected a 3D tensor, got {x.shape}"
            )

        # [B, T, C]
        if x.shape[-1] == 6:

            x = x.transpose(1, 2)

        # Already [B, C, T]
        elif x.shape[1] == 6:

            pass

        else:

            raise ValueError(
                "Input must have shape "
                "[B, 1024, 6] or [B, 6, 1024]. "
                f"Got {x.shape}"
            )

        return x

    # ========================================================
    # Embedding
    # ========================================================

    def forward_embedding(self, x):

        """
        Returns the 1280-dimensional representation
        before the classifier.
        """

        # ----------------------------------------------------
        # Input
        #
        # [B, 1024, 6]
        #
        # -> [B, 6, 1024]
        # ----------------------------------------------------

        x = self._prepare_input(x)

        # ----------------------------------------------------
        # MobileNetV2 feature extractor
        #
        # [B, 6, 1024]
        #
        # -> [B, 1280, T]
        # ----------------------------------------------------

        x = self.features(x)

        # ----------------------------------------------------
        # Global average pooling
        #
        # [B, 1280, T]
        #
        # -> [B, 1280, 1]
        # ----------------------------------------------------

        x = F.adaptive_avg_pool1d(
            x,
            1
        )

        # ----------------------------------------------------
        # Flatten
        #
        # [B, 1280, 1]
        #
        # -> [B, 1280]
        # ----------------------------------------------------

        x = torch.flatten(
            x,
            start_dim=1
        )

        return x

    # ========================================================
    # Forward
    # ========================================================

    def forward(self, x):

        # Get embedding
        embedding = self.forward_embedding(x)

        # Classification
        logits = self.classifier(embedding)

        return logits