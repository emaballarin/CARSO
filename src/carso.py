#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
#  Copyright (c) 2025 Emanuele Ballarin <emanuele@ballarin.cc>
#  Released under the terms of the MIT License
#  (see: https://url.ballarin.cc/mitlicense)
# ──────────────────────────────────────────────────────────────────────────────
import contextlib
from collections.abc import Iterable
from copy import deepcopy
from itertools import chain
from math import ceil
from math import prod as mult
from typing import List
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union

import torch as th
from composer.functional import apply_blurpool
from ebtorch.nn import DuplexLinearNeck
from ebtorch.nn import GaussianReparameterizerSampler
from ebtorch.nn.utils import gather_model_repr
from ebtorch.nn.utils import model_reqgrad_
from ebtorch.nn.utils import tensor_module_matched_apply
from safe_assert import safe_assert as sassert
from tooling.aggregation import select_aggregation
from tooling.modelpieces import classif_decode_ens
from tooling.modelpieces import make_decoder_cifar
from tooling.modelpieces import make_decoder_tiny
from tooling.modelpieces import make_flatcat_compressor
from tooling.modelpieces import make_img_compressor
from tooling.modelpieces import make_lw_repr_compressor
from torch import nn
from torch import Tensor

# ──────────────────────────────────────────────────────────────────────────────
__all__: List[str] = ["CARSOWrap"]


# ──────────────────────────────────────────────────────────────────────────────
T = TypeVar("T", bound="CARSOWrap")


# ──────────────────────────────────────────────────────────────────────────────
class CARSOWrap(nn.Module):
    def __init__(
        self: T,
        wrapped_model: nn.Module,
        input_preproc: nn.Module,
        input_shape: Tuple[int, int, int],
        repr_layers: Tuple[str, ...],
        compr_cond_dim: int,
        joint_latent_dim: int,
        ensemble_size: int,
        differentiable_infer: bool = False,
        agg_method: str = "peel",
        blurpool: bool = False,
        classif_replacement: Optional[nn.Module] = None,
    ) -> None:
        # Validate arguments
        sassert(
            input_shape in ((3, 32, 32), (3, 64, 64)),
            "Unsupported input shape (allowed: 3x32x32, 3x64x64)",
        )
        sassert(compr_cond_dim > 0, "Compressed representation size must be positive")
        sassert(joint_latent_dim > 0, "Compressed joint size must be positive")
        sassert(ensemble_size >= 0, "Ensemble size must be positive (0: deterministic)")
        sassert(
            agg_method in ("logit", "prob", "peel", "count"),
            "Invalid aggregation method",
        )

        super().__init__()

        # ──────────────────────────────────────────────────────────────────────
        # Set dataset config variable
        if input_shape == (3, 32, 32):
            datatiny = False
        else:  # input_shape == (3, 64, 64):
            datatiny = True
        # ──────────────────────────────────────────────────────────────────────

        # Gather representation shapes ephemerally
        fake_model: nn.Module = deepcopy(wrapped_model).to("cpu").eval()

        with th.no_grad():
            fake_y, fake_repr_list = gather_model_repr(fake_model, th.rand(1, *input_shape, device="cpu"), repr_layers)
        repr_shapes = [r.shape[1:] for r in fake_repr_list]
        del fake_model, fake_repr_list

        # Compute representation-dependent sizes
        pre_fcn_repr_size: int = sum([
            mult((
                ceil(c / 8 / int(1 + datatiny)),
                h - 6 - 2 * datatiny,
                w - 6 - 2 * datatiny,
            ))
            for c, h, w in repr_shapes
        ])
        compr_input_size: int = mult((
            4 * input_shape[0],
            ceil(ceil(input_shape[1] / 2) / 2),
            ceil(ceil(input_shape[2] / 2) / 2),
        ))
        pre_fcn_joint_size: int = compr_input_size + compr_cond_dim
        # ──────────────────────────────────────────────────────────────────────

        # Wrapped models
        # ─────────────────────────────────────────────────────
        self.wrapped_model: nn.Module = deepcopy(wrapped_model)

        if classif_replacement is not None:
            self.wrapped_model_final: nn.Module = (
                deepcopy(classif_replacement) if differentiable_infer else classif_replacement
            )
        else:
            self.wrapped_model_final: nn.Module = (
                deepcopy(wrapped_model) if differentiable_infer else self.wrapped_model
            )
        self.input_preproc: nn.Module = deepcopy(input_preproc)
        # ─────────────────────────────────────────────────────
        self.wrapped_model.eval()
        self.wrapped_model.zero_grad(set_to_none=True)
        self.wrapped_model_final.eval()
        self.wrapped_model_final.zero_grad(set_to_none=True)
        self.input_preproc.eval()
        self.input_preproc.zero_grad(set_to_none=True)

        # Subnetworks
        self.input_compressor: nn.Module = make_img_compressor(input_shape[0])
        self.repr_compressors: nn.ModuleList = nn.ModuleList([
            make_lw_repr_compressor(c, compress_more=datatiny) for c, _, _ in repr_shapes
        ])
        self.repr_fcn_compressor: nn.Module = make_flatcat_compressor(pre_fcn_repr_size, compr_cond_dim)
        self.joint_fcn_compressor: nn.Module = make_flatcat_compressor(pre_fcn_joint_size, joint_latent_dim)
        self.neck: nn.Module = DuplexLinearNeck(joint_latent_dim, joint_latent_dim)
        self.sampler: nn.Module = GaussianReparameterizerSampler()

        if not datatiny:
            self.decoder: nn.Module = make_decoder_cifar(joint_latent_dim, compr_cond_dim)
        else:  # datatiny == True:
            self.decoder: nn.Module = make_decoder_tiny(joint_latent_dim, compr_cond_dim)

        self.model_out_shape = fake_y.shape[1:]

        # Cleanup
        del repr_shapes, pre_fcn_repr_size, compr_input_size, pre_fcn_joint_size

        # Eventually apply blurpool surgery
        if blurpool:
            apply_blurpool(self.input_compressor, min_channels=4)
            apply_blurpool(self.repr_compressors, min_channels=0)
            apply_blurpool(self.decoder, min_channels=0)

        # ──────────────────────────────────────────────────────────────────────
        # Inference setup
        self.infer_sampler = th.zeros if ensemble_size == 0 else th.randn
        ensemble_size = max(1, ensemble_size)
        self.agg: str = agg_method
        # ──────────────────────────────────────────────────────────────────────
        self.repr_layers = repr_layers
        self.di = differentiable_infer
        self.ensize = ensemble_size
        self.jld = joint_latent_dim

    def train(self: T, mode: bool = True) -> T:
        super().train(mode)
        # ──────────────────────────────────────────────────────────────────────
        m_or_ndi = mode or not self.di

        self.wrapped_model.train(False)
        self.input_preproc.train(False)
        self.wrapped_model_final.train(False)
        self.wrapped_model.zero_grad(set_to_none=m_or_ndi)
        self.input_preproc.zero_grad(set_to_none=m_or_ndi)
        self.wrapped_model_final.zero_grad(set_to_none=m_or_ndi)
        model_reqgrad_(self.wrapped_model, not m_or_ndi)
        model_reqgrad_(self.wrapped_model_final, not m_or_ndi)
        model_reqgrad_(self.input_preproc, not m_or_ndi)
        # ──────────────────────────────────────────────────────────────────────
        return self

    def named_parameters(self: T, *args, **kwargs) -> Iterable[Tuple[str, nn.Parameter]]:
        yield from chain(
            self.input_compressor.named_parameters(*args, **kwargs),
            self.repr_compressors.named_parameters(*args, **kwargs),
            self.repr_fcn_compressor.named_parameters(*args, **kwargs),
            self.joint_fcn_compressor.named_parameters(*args, **kwargs),
            self.neck.named_parameters(*args, **kwargs),
            self.sampler.named_parameters(*args, **kwargs),
            self.decoder.named_parameters(*args, **kwargs),
        )

    def forward(self: T, x: Tensor) -> Union[Tuple[Tensor, Tensor, Tensor], Tensor]:
        # ──────────────────────────────────────────────────────────────────────
        if self.training:
            with th.no_grad():
                x.requires_grad_(False)
                _, repr_list = gather_model_repr(self.wrapped_model, x, self.repr_layers, preserve_graph=False)
                del _

            x.requires_grad_(True)
            for r in repr_list:
                r.requires_grad_(True)

            x: Tensor = self.input_preproc(x)
            x: Tensor = self.input_compressor(x)

            repr_list: List[Tensor] = tensor_module_matched_apply(repr_list, self.repr_compressors)
            latent_c: Tensor = self.repr_fcn_compressor(repr_list)

            latent_z: Tensor = self.joint_fcn_compressor((x, latent_c))

            mu, logvar = self.neck((latent_z,))
            zsample: Tensor = self.sampler(mu, logvar)

            xrec: Tensor = self.decoder((zsample, latent_c))

            return xrec, mu, logvar

        # ──────────────────────────────────────────────────────────────────────
        else:
            # ──────────────────────────────────────────────────────────────────
            with contextlib.ExitStack() as stack:
                if self.di:
                    stack.enter_context(th.enable_grad())
                    x.requires_grad_(True)
                else:
                    stack.enter_context(th.no_grad())
                    x = x.detach()
                # ──────────────────────────────────────────────────────────────
                _, repr_list = gather_model_repr(self.wrapped_model, x, self.repr_layers, preserve_graph=self.di)
                del _

                repr_list: List[Tensor] = tensor_module_matched_apply(repr_list, self.repr_compressors)
                latent_c: Tensor = self.repr_fcn_compressor(repr_list)
                # ──────────────────────────────────────────────────────────────
                samples = self.infer_sampler((x.shape[0], self.jld, self.ensize), device=x.device)
                out: Tensor = classif_decode_ens(self.wrapped_model_final, self.decoder, samples, latent_c)
                return select_aggregation(method=self.agg)(out)
