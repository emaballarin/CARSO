#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Optional

from ebtorch.nn import FCBlock
from ebtorch.nn import FieldTransform
from ebtorch.nn import KWTA1d
from torch.nn import Flatten
from torch.nn import functional as F
from torch.nn import LeakyReLU
from torch.nn import Linear
from torch.nn import LogSoftmax
from torch.nn import Mish
from torch.nn import ModuleList
from torch.nn import Sequential
from torch.nn import Sigmoid
from torch.nn import Tanh


# ---- UTILITY FUNCTIONS ----
def pixelwise_bce_sum(lhs, rhs):
    return F.binary_cross_entropy(lhs, rhs, reduction="sum")


def pixelwise_bce_mean(lhs, rhs):
    return F.binary_cross_entropy(lhs, rhs, reduction="mean")


# ---- MNIST FCN CLASSIFIER DIspatcher ----
def mnistfcn_dispatcher(device=None, use_kwta=False):
    mnistfcn = Sequential(
        FieldTransform(pre_sum=-0.1307, mult_div=0.3081, div_not_mul=True),
        Flatten(),
        FCBlock(
            in_sizes=(28 * 28, 200, 80),
            out_size=10,
            bias=True,
            activation_fx=(
                ModuleList(modules=(Mish(), Mish(), Mish()))
                if not use_kwta
                else ModuleList(
                    modules=(
                        KWTA1d(largest=True, absolute=True, ratio=0.34),
                        KWTA1d(largest=True, absolute=True, ratio=0.25),
                        KWTA1d(largest=True, absolute=True, ratio=0.2),
                    )
                )
            ),
            dropout=([0.15, 0.15, False] if not use_kwta else [False, False, False]),
            batchnorm=[True, True, False],
        ),
        LogSoftmax(dim=1),
    )

    if device is not None:
        mnistfcn = mnistfcn.to(device)

    return mnistfcn


def mnist_data_prep_dispatcher(device=None):
    mnist_data_prep = Sequential(
        FieldTransform(pre_sum=-0.1307, mult_div=0.3081, div_not_mul=True), Flatten()
    )

    if device is not None:
        mnist_data_prep = mnist_data_prep.to(device)

    return mnist_data_prep


# ---- REPRESENTATION COMPRESSOR ----
def compressor_dispatcher(input_size: int, compress_size: int, device=None):

    compress = Sequential(
        FCBlock(
            in_sizes=(input_size, (input_size + compress_size) // 2),
            out_size=compress_size,
            bias=True,
            activation_fx=ModuleList(modules=(LeakyReLU(), LeakyReLU())),
            dropout=False,
            batchnorm=[True, True],
        ),
        Sigmoid(),  # Prepare for VIM-VAE already
    )

    if device is not None:
        compress = compress.to(device)

    return compress


# ---- VIM (C)VAE ----
def fcn_carso_dispatcher(
    random_variable_size: int,
    conditioning_set_size: int,
    precompress_size: int,
    compress_size: int,
    actual_output_size: Optional[int] = None,
    device=None,
):

    # Household tasks
    if actual_output_size is None:
        actual_output_size = random_variable_size

    # "Encoder"
    carso_enc_neck = Sequential(
        FCBlock(
            in_sizes=(
                random_variable_size + conditioning_set_size,
                (random_variable_size + conditioning_set_size + precompress_size) // 2,
            ),
            out_size=precompress_size,
            bias=True,
            activation_fx=ModuleList(modules=(LeakyReLU(), LeakyReLU())),
            dropout=[0.075 / 2, False],
            batchnorm=[True, True],
        ),
        Tanh(),
    )
    carso_enc_mu = Linear(
        in_features=precompress_size, out_features=compress_size, bias=True
    )
    carso_enc_sigma = Linear(
        in_features=precompress_size, out_features=compress_size, bias=True
    )

    # "Decoder"
    carso_dec = Sequential(
        FCBlock(
            in_sizes=(
                compress_size + conditioning_set_size,
                int(
                    (
                        actual_output_size
                        + 2
                        * (
                            abs(
                                compress_size
                                + conditioning_set_size
                                - actual_output_size
                            )
                        )
                        / 3
                    )
                ),
                int(
                    (
                        actual_output_size
                        + (
                            abs(
                                compress_size
                                + conditioning_set_size
                                - actual_output_size
                            )
                        )
                        / 3
                    )
                ),
            ),
            out_size=actual_output_size,
            bias=True,
            activation_fx=ModuleList(modules=(LeakyReLU(), LeakyReLU(), LeakyReLU())),
            dropout=[0.075, 0.075 / 2, False],
            batchnorm=[True, True, False],
        ),
        Sigmoid(),
    )

    # Device handling
    if device is not None:
        carso_enc_neck, carso_enc_mu, carso_enc_sigma, carso_dec = (
            carso_enc_neck.to(device),
            carso_enc_mu.to(device),
            carso_enc_sigma.to(device),
            carso_dec.to(device),
        )

    return carso_enc_neck, carso_enc_mu, carso_enc_sigma, carso_dec
