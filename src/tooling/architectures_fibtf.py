#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  Copyright (c) 2023 Emanuele Ballarin <emanuele@ballarin.cc>
#  Released under the terms of the MIT License
#  (see: https://url.ballarin.cc/mitlicense)
#
# ------------------------------------------------------------------------------
from typing import List

import torch as th
import torch.nn as thnn
import torch.nn.functional as F
from ebtorch import nn as ebthnn


class PreActBlock(thnn.Module):
    """Pre-activation version of the BasicBlock."""

    expansion: int = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super(PreActBlock, self).__init__()
        self.bn1: thnn.Module = thnn.BatchNorm2d(in_planes)
        self.conv1: thnn.Module = thnn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2: thnn.Module = thnn.BatchNorm2d(planes)
        self.conv2: thnn.Module = thnn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut: thnn.Module = thnn.Sequential(
                thnn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                )
            )

    def forward(self, x: th.Tensor) -> th.Tensor:
        out: th.Tensor = F.relu(self.bn1(x))
        shortcut: th.Tensor = self.shortcut(x) if hasattr(self, "shortcut") else x
        out: th.Tensor = self.conv1(out)
        out: th.Tensor = self.conv2(F.relu(self.bn2(out)))
        out: th.Tensor = out + shortcut
        return out


class PreActBottleneck(thnn.Module):
    """Pre-activation version of the original Bottleneck module."""

    expansion: int = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super(PreActBottleneck, self).__init__()
        self.bn1: thnn.Module = thnn.BatchNorm2d(in_planes)
        self.conv1: thnn.Module = thnn.Conv2d(
            in_planes, planes, kernel_size=1, bias=False
        )
        self.bn2: thnn.Module = thnn.BatchNorm2d(planes)
        self.conv2: thnn.Module = thnn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn3: thnn.Module = thnn.BatchNorm2d(planes)
        self.conv3: thnn.Module = thnn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut: thnn.Module = thnn.Sequential(
                thnn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                )
            )

    def forward(self, x: th.Tensor) -> th.Tensor:
        out: th.Tensor = F.relu(self.bn1(x))
        shortcut: th.Tensor = self.shortcut(out) if hasattr(self, "shortcut") else x
        out: th.Tensor = self.conv1(out)
        out: th.Tensor = self.conv2(F.relu(self.bn2(out)))
        out: th.Tensor = self.conv3(F.relu(self.bn3(out)))
        out: th.Tensor = out + shortcut
        return out


class PreActResNet(thnn.Module):
    def __init__(self, block, num_blocks: List[int], num_classes: int = 10) -> None:
        super(PreActResNet, self).__init__()
        self.in_planes: int = 64
        self.conv1: thnn.Module = thnn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.layer1: thnn.Module = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2: thnn.Module = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3: thnn.Module = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4: thnn.Module = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn: thnn.Module = thnn.BatchNorm2d(512 * block.expansion)
        self.linear: thnn.Module = thnn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self, block: thnn.Module, planes: int, num_blocks: int, stride: int
    ) -> thnn.Module:
        strides: List[int] = [stride] + [1] * (num_blocks - 1)
        layers: List[thnn.Module] = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes: int = planes * block.expansion
        return thnn.Sequential(*layers)

    def forward(self, x: th.Tensor) -> th.Tensor:
        out: th.Tensor = self.conv1(x)
        out: th.Tensor = self.layer1(out)
        out: th.Tensor = self.layer2(out)
        out: th.Tensor = self.layer3(out)
        out: th.Tensor = self.layer4(out)
        out: th.Tensor = F.relu(self.bn(out))
        out: th.Tensor = F.avg_pool2d(out, 4)
        out: th.Tensor = out.view(out.size(0), -1)
        out: th.Tensor = self.linear(out)
        return out


class PreActResNet18Cifar10(thnn.Module):
    def __init__(self, device) -> None:
        super(PreActResNet18Cifar10, self).__init__()
        self.model: thnn.Module = PreActResNet(PreActBlock, [2, 2, 2, 2])
        self.dataprep: thnn.Module = ebthnn.FieldTransform(
            pre_sum=th.tensor([[[-0.4914]], [[-0.4822]], [[-0.4465]]]).to(device),
            mult_div=th.tensor([[[0.2471]], [[0.2435]], [[0.2616]]]).to(device),
            div_not_mul=True,
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        x: th.Tensor = self.dataprep(x)
        x: th.Tensor = self.model(x)
        return x
