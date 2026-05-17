import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _channel_stats(values, reference_tensor):
    if torch.is_tensor(values):
        values = values.detach().to(dtype=reference_tensor.dtype, device=reference_tensor.device)
    else:
        values = torch.tensor(values, dtype=reference_tensor.dtype, device=reference_tensor.device)

    shape = (1, -1, 1, 1) if reference_tensor.ndim == 4 else (-1, 1, 1)
    return values.flatten().view(*shape)


def denormalize(tensor, mean=DEFAULT_MEAN, std=DEFAULT_STD):
    mean_tensor = _channel_stats(mean, tensor)
    std_tensor = _channel_stats(std, tensor)
    return (tensor * std_tensor + mean_tensor).clamp(0, 1)


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def make_group_norm(num_channels, max_groups=8):
    groups = min(max_groups, num_channels)
    while groups > 1 and (num_channels % groups != 0 or num_channels // groups < 2):
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, dilation=1):
        if padding is None:
            padding = dilation * (kernel_size // 2)
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            make_group_norm(out_channels),
            nn.ReLU(inplace=True),
        )


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(in_channels, out_channels, dilation=dilation),
            ConvNormAct(out_channels, out_channels, dilation=dilation),
        )

    def forward(self, x):
        return self.block(x)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=128, rates=(1, 6, 12, 18), dropout=0.1):
        super().__init__()
        branches = []
        for rate in rates:
            if rate == 1:
                branches.append(ConvNormAct(in_channels, out_channels, kernel_size=1, padding=0))
            else:
                branches.append(ConvNormAct(in_channels, out_channels, kernel_size=3, dilation=rate))
        self.branches = nn.ModuleList(branches)
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvNormAct(in_channels, out_channels, kernel_size=1, padding=0),
        )
        self.project = nn.Sequential(
            ConvNormAct(out_channels * (len(rates) + 1), out_channels, kernel_size=1, padding=0),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        size = x.shape[-2:]
        outputs = [branch(x) for branch in self.branches]
        pooled = self.image_pool(x)
        pooled = F.interpolate(pooled, size=size, mode="bilinear", align_corners=False)
        outputs.append(pooled)
        return self.project(torch.cat(outputs, dim=1))


class DeepLabV3Plus(nn.Module):
    def __init__(
        self,
        num_classes=3,
        base_channels=32,
        aspp_channels=128,
        decoder_channels=128,
        low_channels=48,
        rates=(1, 6, 12, 18),
    ):
        super().__init__()
        self.stem = DoubleConv(3, base_channels)
        self.stage2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channels, base_channels * 2))
        self.stage3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channels * 2, base_channels * 4))
        self.stage4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_channels * 4, base_channels * 8),
            ConvNormAct(base_channels * 8, base_channels * 8, dilation=2),
            ConvNormAct(base_channels * 8, base_channels * 8, dilation=4),
        )
        self.aspp = ASPP(base_channels * 8, aspp_channels, rates=rates)
        self.low_project = ConvNormAct(base_channels * 2, low_channels, kernel_size=1, padding=0)
        self.decoder = nn.Sequential(
            DoubleConv(aspp_channels + low_channels, decoder_channels),
            nn.Conv2d(decoder_channels, num_classes, kernel_size=1),
        )

    def forward(self, x):
        input_size = x.shape[-2:]
        x = self.stem(x)
        low = self.stage2(x)
        x = self.stage3(low)
        high = self.stage4(x)

        high = self.aspp(high)
        high = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        low = self.low_project(low)
        logits = self.decoder(torch.cat([low, high], dim=1))
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
