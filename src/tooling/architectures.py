#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  Copyright (c) 2023 Emanuele Ballarin <emanuele@ballarin.cc>
#  Released under the terms of the MIT License
#  (see: https://url.ballarin.cc/mitlicense)
#
# ------------------------------------------------------------------------------
from copy import deepcopy
from typing import Optional
from typing import Tuple

from ebtorch import nn as ebthnn
from torch import nn as thnn

from .architectures_fibtf import PreActResNet18Cifar10


# (De)convolutional blocks
def _conv_block(in_channels, out_channels, kernel_size, stride, padding, final=False):
    if final:
        final_activation = thnn.Sigmoid()
    else:
        final_activation = thnn.LeakyReLU()
    return thnn.Sequential(
        thnn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,  # BatchNorm2d will add "bias" nonetheless
        ),
        thnn.BatchNorm2d(num_features=out_channels),
        deepcopy(final_activation),
    )


def _deconv_block(
    in_channels, out_channels, kernel_size, stride, padding, final=False, do_bn=True
):
    if final:
        final_activation = thnn.Sigmoid()
    else:
        final_activation = thnn.LeakyReLU()
    return thnn.Sequential(
        thnn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=not do_bn,  # BatchNorm2d will add "bias" nonetheless
        ),
        thnn.BatchNorm2d(num_features=out_channels) if do_bn else thnn.Identity(),
        deepcopy(final_activation),
    )


# Classifier models


def mnist_fcn_classifier_dispatcher(kwta_filter=False) -> thnn.Module:
    mnistfcn = thnn.Sequential(
        ebthnn.FieldTransform(pre_sum=-0.1307, mult_div=0.3081, div_not_mul=True),
        thnn.Flatten(),
        ebthnn.FCBlock(
            in_sizes=(28 * 28, 200, 80),
            out_size=10,
            bias=True,
            activation_fx=(
                thnn.ModuleList(modules=(thnn.Mish(), thnn.Mish(), thnn.Mish()))
                if not kwta_filter
                else thnn.ModuleList(
                    modules=(
                        thnn.Sequential(
                            thnn.Mish(),
                            ebthnn.KWTA1d(largest=True, absolute=True, ratio=0.3),
                        ),
                        thnn.Sequential(
                            thnn.Mish(),
                            ebthnn.KWTA1d(largest=True, absolute=True, ratio=0.3),
                        ),
                        thnn.Mish(),
                    )
                )
            ),
            dropout=([0.15, 0.15, False] if not kwta_filter else [False, False, False]),
            batchnorm=[True, True, False],
        ),
        thnn.LogSoftmax(dim=1),
    )

    return mnistfcn


def mnist_cnn_classifier_dispatcher() -> thnn.Module:
    mnistcnn = thnn.Sequential(
        ebthnn.FieldTransform(pre_sum=-0.1307, mult_div=0.3081, div_not_mul=True),
        thnn.Conv2d(1, 16, 4, stride=2, padding=1),
        thnn.Mish(),
        thnn.BatchNorm2d(num_features=16),
        thnn.Conv2d(16, 32, 4, stride=2, padding=1),
        thnn.Mish(),
        thnn.BatchNorm2d(num_features=32),
        thnn.Flatten(),
        thnn.Dropout(p=0.1),
        thnn.Linear(32 * 7 * 7, 100),
        thnn.Mish(),
        thnn.BatchNorm1d(num_features=100),
        thnn.Dropout(p=0.1),
        thnn.Linear(100, 10),
        thnn.LogSoftmax(dim=1),
    )
    return mnistcnn


def fashionmnist_cnn_classifier_dispatcher() -> thnn.Module:
    mnistcnn = thnn.Sequential(
        thnn.Conv2d(1, 16, 4, stride=2, padding=1),
        thnn.Mish(),
        thnn.BatchNorm2d(num_features=16),
        thnn.Conv2d(16, 32, 4, stride=2, padding=1),
        thnn.Mish(),
        thnn.BatchNorm2d(num_features=32),
        thnn.Flatten(),
        thnn.Dropout(p=0.1),
        thnn.Linear(32 * 7 * 7, 100),
        thnn.Mish(),
        thnn.BatchNorm1d(num_features=100),
        thnn.Dropout(p=0.1),
        thnn.Linear(100, 10),
        thnn.LogSoftmax(dim=1),
    )
    return mnistcnn


def cifarten_resnet18_classifier_dispatcher(device) -> thnn.Module:
    return PreActResNet18Cifar10(device).to(device)


# Compressor models


def fcn_compressor_dispatcher(
    input_size: int, output_size: int, slim_neck: bool = False
) -> thnn.Module:
    neck_size = 2 * output_size if slim_neck else (input_size + output_size) // 2
    compressor = thnn.Sequential(
        thnn.Flatten(),
        ebthnn.FCBlock(
            in_sizes=(input_size, neck_size),
            out_size=output_size,
            bias=True,
            activation_fx=thnn.ModuleList(modules=(thnn.LeakyReLU(), thnn.LeakyReLU())),
            dropout=False,
            batchnorm=[True, True],
        ),
        thnn.Sigmoid(),
    )
    return compressor


def cnn_compressor_dispatcher_flatout(channels_in, channels_out):
    compressor = thnn.Sequential(
        _conv_block(channels_in, channels_out // 4, 4, 2, 1, final=False),
        _conv_block(channels_out // 4, channels_out // 2, 4, 2, 1, final=False),
        _conv_block(channels_out // 2, channels_out, 4, 2, 1, final=True),
        thnn.Flatten(),
    )
    return compressor


# Encoder-decoder models


def encdec_dispatcher(
    data_size: int,
    condition_size: int,
    shared_musigma_layer_size: int,
    sampled_code_size: int,
    output_size: Optional[int] = None,
    input_channels: Optional[int] = None,
    deconvolutional: bool = False,
    cifar: bool = True,
) -> Tuple[thnn.Module, thnn.Module, thnn.Module, thnn.Module]:
    if output_size is None:
        output_size = data_size + condition_size

    if input_channels is None:
        input_channels = data_size + condition_size

    # Encoder
    enc_neck: thnn.Module = thnn.Sequential(
        ebthnn.FCBlock(
            in_sizes=(
                data_size + condition_size,
                (data_size + condition_size + shared_musigma_layer_size) // 2,
            ),
            out_size=shared_musigma_layer_size,
            bias=True,
            activation_fx=thnn.ModuleList(modules=(thnn.LeakyReLU(), thnn.LeakyReLU())),
            dropout=[0.075 / 2, False],
            batchnorm=[True, True],
        ),
        thnn.Tanh(),
    )
    enc_mu: thnn.Module = thnn.Linear(
        in_features=shared_musigma_layer_size, out_features=sampled_code_size, bias=True
    )
    enc_sigma: thnn.Module = thnn.Linear(
        in_features=shared_musigma_layer_size, out_features=sampled_code_size, bias=True
    )

    # Decoder
    if deconvolutional:
        del output_size
        if cifar:
            # CIFAR-10/100
            dec: thnn.Module = thnn.Sequential(
                ebthnn.FlatChannelize2DLayer(),
                _deconv_block(
                    in_channels=input_channels,
                    out_channels=input_channels // 2,
                    kernel_size=4,
                    stride=1,
                    padding=0,
                    final=False,
                ),
                _deconv_block(
                    in_channels=input_channels // 2,
                    out_channels=input_channels // 4,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    final=False,
                ),
                _deconv_block(
                    in_channels=input_channels // 4,
                    out_channels=input_channels // 8,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    final=False,
                ),
                _deconv_block(
                    in_channels=input_channels // 8,
                    out_channels=3,  # RGB
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    final=True,
                ),
            )
        else:
            # MNIST/Fashion-MNIST
            dec: thnn.Module = thnn.Sequential(
                ebthnn.FlatChannelize2DLayer(),
                _deconv_block(
                    in_channels=input_channels,
                    out_channels=input_channels // 2,
                    kernel_size=4,
                    stride=1,
                    padding=0,
                    final=False,
                    do_bn=False,
                ),
                _deconv_block(
                    in_channels=input_channels // 2,
                    out_channels=input_channels // 4,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    final=False,
                    do_bn=False,
                ),
                _deconv_block(
                    in_channels=input_channels // 4,
                    out_channels=input_channels // 8,
                    kernel_size=4,
                    stride=2,
                    padding=2,
                    final=False,
                    do_bn=False,
                ),
                _deconv_block(
                    in_channels=input_channels // 8,
                    out_channels=1,  # Grayscale
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    final=True,
                    do_bn=False,
                ),
            )
    else:
        del input_channels
        dec: thnn.Module = thnn.Sequential(
            ebthnn.FCBlock(
                in_sizes=(
                    sampled_code_size + condition_size,
                    int(
                        (
                            output_size
                            + 2
                            * (abs(sampled_code_size + condition_size - output_size))
                            / 3
                        )
                    ),
                    int(
                        (
                            output_size
                            + (abs(sampled_code_size + condition_size - output_size))
                            / 3
                        )
                    ),
                ),
                out_size=output_size,
                bias=True,
                activation_fx=thnn.ModuleList(
                    modules=(thnn.LeakyReLU(), thnn.LeakyReLU(), thnn.LeakyReLU())
                ),
                dropout=[0.075, 0.075 / 2, False],
                batchnorm=[True, True, False],
            ),
            thnn.Sigmoid(),
        )

    return enc_neck, enc_mu, enc_sigma, dec
