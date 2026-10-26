from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ASPP(nn.Module):
    """ASPP at the U-Net++ bottleneck."""

    def __init__(self, channels: int, rates: Iterable[int] = (3, 6, 9)) -> None:
        super().__init__()
        branches = [
            nn.Sequential(
                nn.Conv2d(channels, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
        ]
        for rate in rates:
            branches.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=rate, dilation=rate, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
            )
        self.branches = nn.ModuleList(branches)
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1, bias=False),
            # GroupNorm remains valid when the pooled feature map is 1x1 and
            # the last training batch contains a single image.
            nn.GroupNorm(1, channels),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(channels * (len(branches) + 1), channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        features = [branch(x) for branch in self.branches]
        pooled = self.pool(x)
        pooled = F.interpolate(pooled, size=(height, width), mode="bilinear", align_corners=False)
        return self.project(torch.cat(features + [pooled], dim=1))


class SqueezeExcitation(nn.Module):
    """The standard squeeze-and-excitation channel recalibration block."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.project(self.pool(x))


class CBAM(nn.Module):
    """Convolutional block attention module with channel and spatial attention."""

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.channel_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_max = nn.AdaptiveMaxPool2d(1)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.channel_sigmoid = nn.Sigmoid()
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel = self.channel_sigmoid(self.channel_mlp(self.channel_pool(x)) + self.channel_mlp(self.channel_max(x)))
        x = x * channel
        spatial = self.spatial(torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1))
        return x * spatial


class UNetPlusPlus(nn.Module):
    """U-Net++: bottleneck ASPP, final decoder SE -> CBAM -> head."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 32,
        use_aspp: bool = False,
        use_se: bool = False,
        use_cbam: bool = False,
    ) -> None:
        super().__init__()
        channels = [base_channels * (2**i) for i in range(5)]
        self.pool = nn.MaxPool2d(2, 2)

        self.x00 = ConvBlock(in_channels, channels[0])
        self.x10 = ConvBlock(channels[0], channels[1])
        self.x20 = ConvBlock(channels[1], channels[2])
        self.x30 = ConvBlock(channels[2], channels[3])
        self.x40 = ConvBlock(channels[3], channels[4])

        self.x01 = ConvBlock(channels[0] + channels[1], channels[0])
        self.x11 = ConvBlock(channels[1] + channels[2], channels[1])
        self.x21 = ConvBlock(channels[2] + channels[3], channels[2])
        self.x31 = ConvBlock(channels[3] + channels[4], channels[3])
        self.x02 = ConvBlock(channels[0] * 2 + channels[1], channels[0])
        self.x12 = ConvBlock(channels[1] * 2 + channels[2], channels[1])
        self.x22 = ConvBlock(channels[2] * 2 + channels[3], channels[2])
        self.x03 = ConvBlock(channels[0] * 3 + channels[1], channels[0])
        self.x13 = ConvBlock(channels[1] * 3 + channels[2], channels[1])
        self.x04 = ConvBlock(channels[0] * 4 + channels[1], channels[0])

        self.aspp = ASPP(channels[4]) if use_aspp else nn.Identity()
        self.se = SqueezeExcitation(channels[0]) if use_se else nn.Identity()
        self.cbam = CBAM(channels[0]) if use_cbam else nn.Identity()
        self.head = nn.Conv2d(channels[0], num_classes, 1)

    @staticmethod
    def upsample(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x00 = self.x00(x)
        x10 = self.x10(self.pool(x00))
        x20 = self.x20(self.pool(x10))
        x30 = self.x30(self.pool(x20))
        x40 = self.aspp(self.x40(self.pool(x30)))

        x01 = self.x01(torch.cat([x00, self.upsample(x10, x00)], dim=1))
        x11 = self.x11(torch.cat([x10, self.upsample(x20, x10)], dim=1))
        x21 = self.x21(torch.cat([x20, self.upsample(x30, x20)], dim=1))
        x31 = self.x31(torch.cat([x30, self.upsample(x40, x30)], dim=1))

        x02 = self.x02(torch.cat([x00, x01, self.upsample(x11, x00)], dim=1))
        x12 = self.x12(torch.cat([x10, x11, self.upsample(x21, x10)], dim=1))
        x22 = self.x22(torch.cat([x20, x21, self.upsample(x31, x20)], dim=1))

        x03 = self.x03(torch.cat([x00, x01, x02, self.upsample(x12, x00)], dim=1))
        x13 = self.x13(torch.cat([x10, x11, x12, self.upsample(x22, x10)], dim=1))
        x04 = self.x04(torch.cat([x00, x01, x02, x03, self.upsample(x13, x00)], dim=1))

        x04 = self.se(x04)
        x04 = self.cbam(x04)
        return self.head(x04)


MODEL_FLAGS = {
    # (ASPP, SE, CBAM)
    "baseline": (False, False, False),
    "aspp": (True, False, False),
    "se": (False, True, False),
    "cbam": (False, False, True),
    "aspp_se": (True, True, False),
    "aspp_cbam": (True, False, True),
    "cbam_se": (False, True, True),
    "aspp_cbam_se": (True, True, True),
}


def build_model(name: str, **kwargs: object) -> UNetPlusPlus:
    if name not in MODEL_FLAGS:
        raise ValueError(f"Unknown model: {name}. Choices: {list(MODEL_FLAGS)}")
    aspp, se, cbam = MODEL_FLAGS[name]
    return UNetPlusPlus(use_aspp=aspp, use_se=se, use_cbam=cbam, **kwargs)
