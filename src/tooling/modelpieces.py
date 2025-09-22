#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
#  Copyright (c) 2025 Emanuele Ballarin <emanuele@ballarin.cc>
#  Released under the terms of the MIT License
#  (see: https://url.ballarin.cc/mitlicense)
# ──────────────────────────────────────────────────────────────────────────────
from math import ceil
from math import prod as mult
from typing import List
from typing import Tuple
from typing import Union

import torch as th
from ebtorch.nn import Concatenate
from torch import nn
from torch import Tensor

# ──────────────────────────────────────────────────────────────────────────────
__all__: List[str] = [
    "make_lw_repr_compressor",
    "make_img_compressor",
    "make_flatcat_compressor",
    "make_decoder_cifar",
    "make_decoder_tiny",
    "classif_decode_ens",
]


# ──────────────────────────────────────────────────────────────────────────────
def _classif_decode_atomic(
    classifier: nn.Module,
    decoder: nn.Module,
    zsample: Tensor,
    latent_c: Tensor,
):
    return classifier(decoder((zsample, latent_c)))


classif_decode_ens = th.vmap(
    _classif_decode_atomic,
    in_dims=(None, None, -1, None),
    out_dims=-1,
)
# ──────────────────────────────────────────────────────────────────────────────


def _make_conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    leaky_slope: float = 0.2,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_channels, affine=True),
        nn.LeakyReLU(leaky_slope),
    )


def _make_fc_block(in_features: int, out_features: int, leaky_slope: float = 0.2) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=False),
        nn.BatchNorm1d(out_features, affine=True),
        nn.LeakyReLU(leaky_slope),
    )


# ──────────────────────────────────────────────────────────────────────────────


def _make_deconv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    output_padding: Union[int, Tuple[int, int]] = 0,
    leaky_slope: float = 0.2,
) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels, affine=True),
        nn.LeakyReLU(leaky_slope),
    )


def _make_decoder_adapter(
    lat_size: int,
    cond_size: int,
    target_shape: Tuple[int, int, int],
    leaky_slope: float = 0.2,
) -> nn.Sequential:
    return nn.Sequential(
        Concatenate(flatten=True),
        nn.Linear(lat_size + cond_size, mult(target_shape)),
        nn.LeakyReLU(leaky_slope),
        nn.Unflatten(1, target_shape),
    )


# ──────────────────────────────────────────────────────────────────────────────


def make_lw_repr_compressor(in_channels: int, compress_more: bool = False) -> nn.Sequential:
    return nn.Sequential(
        _make_conv_block(in_channels, ceil(in_channels / 2), 3, 1, 0),
        _make_conv_block(ceil(in_channels / 2), ceil(in_channels / 4), 3, 1, 0),
        _make_conv_block(ceil(in_channels / 4), ceil(in_channels / 8), 3, 1, 0),
        (_make_conv_block(ceil(in_channels / 8), ceil(in_channels / 16), 3, 1, 0) if compress_more else nn.Identity()),
    )


def make_lw_repr_compressor_l(in_feats: int, out_feats: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Flatten(start_dim=1),
        nn.Linear(in_feats, out_feats, bias=False),
        nn.BatchNorm1d(out_feats, affine=True),
        nn.LeakyReLU(0.2),
    )


def make_img_compressor(in_channels: int) -> nn.Sequential:
    return nn.Sequential(
        _make_conv_block(in_channels, 2 * in_channels, 3, 2, 1),
        _make_conv_block(2 * in_channels, 4 * in_channels, 3, 2, 1),
    )


def make_flatcat_compressor(
    in_features: int,
    out_features: int,
) -> nn.Sequential:
    return nn.Sequential(
        Concatenate(flatten=True),
        _make_fc_block(in_features, out_features),
    )


# ──────────────────────────────────────────────────────────────────────────────


def make_decoder_cifar(lat_size: int, cond_size: int) -> nn.Sequential:
    return nn.Sequential(
        _make_decoder_adapter(lat_size, cond_size, (256, 3, 3)),
        _make_deconv_block(256, 256, 3, 2, 1, 0),
        _make_deconv_block(256, 128, 3, 2, 1, 0),
        _make_deconv_block(128, 64, 3, 2, 1, 0),
        _make_deconv_block(64, 32, 3, 2, 1, 0),
        nn.ConvTranspose2d(32, 3, 2, 1, 1, 0, bias=True),
        nn.Sigmoid(),
    )


def make_decoder_tiny(lat_size: int, cond_size: int) -> nn.Sequential:
    return nn.Sequential(
        _make_decoder_adapter(lat_size, cond_size, (256, 4, 4)),
        _make_deconv_block(256, 256, 3, 2, 1, 1),
        _make_deconv_block(256, 128, 3, 2, 1, 1),
        _make_deconv_block(128, 64, 3, 2, 1, 1),
        _make_deconv_block(64, 32, 3, 2, 1, 1),
        nn.ConvTranspose2d(32, 3, 3, 1, 1, 0, bias=True),
        nn.Sigmoid(),
    )
