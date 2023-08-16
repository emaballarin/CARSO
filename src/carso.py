#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  Copyright (c) 2023 Emanuele Ballarin <emanuele@ballarin.cc>
#  Released under the terms of the MIT License
#  (see: https://url.ballarin.cc/mitlicense)
#
# ------------------------------------------------------------------------------
import contextlib
import copy
import itertools
import warnings
from collections.abc import Callable
from collections.abc import Iterable
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union

import torch as th
from ebtorch import nn as ebthnn
from tooling.architectures import cnn_compressor_dispatcher_flatout
from tooling.architectures import encdec_dispatcher
from tooling.architectures import fcn_compressor_dispatcher
from torch import nn as thnn


T = TypeVar("T", bound="CARSOWrap")
# ------------------------------------------------------------------------------


def carso_softmax_harden_max(x: th.Tensor) -> th.Tensor:
    return th.floor(x / x.max(dim=2, keepdim=True)[0])


def carso_softmax_identity(x: th.Tensor) -> th.Tensor:
    return x


# ------------------------------------------------------------------------------


def carso_ensembled_classifier_atomic(
    eca_ensembled_classifier: th.nn.Module,
    eca_dec: th.nn.Module,
    eca_compressed_repr: th.Tensor,
    eca_compressed_repr_shape_zero: int,
    eca_compressed_size: int,
    final_shape: Tuple,
    suppress_randomness: bool = False,
) -> th.Tensor:
    if suppress_randomness:
        sampler = th.zeros
    else:
        sampler = th.randn

    return eca_ensembled_classifier(
        eca_dec(
            th.cat(
                (
                    sampler(
                        (
                            eca_compressed_repr_shape_zero,
                            eca_compressed_size,
                        ),
                        device=eca_compressed_repr.device,
                    ),
                    eca_compressed_repr,
                ),
                dim=1,
            )
        ).reshape(*final_shape)
    )


carso_ensembled_classifier_ensemblebatched = th.vmap(
    func=carso_ensembled_classifier_atomic,
    in_dims=(None, None, 0, None, None, None, None),
    randomness="different",
)

# ------------------------------------------------------------------------------


class CARSOWrap(thnn.Module):
    def __init__(
        self: T,
        wrapped_model: thnn.Module,
        input_data_height: int,
        input_data_width: int,
        input_data_channels: int,
        wrapped_repr_size: int,
        compressed_input_data_size: int,
        compressed_repr_data_size: int,
        shared_musigma_layer_size: int,
        sampled_code_size: int,
        ensemble_numerosity: int,
        input_data_no_compress: bool = False,
        input_data_conv_flatten: bool = False,
        convolutional_input_compressor: bool = False,
        repr_data_no_compress: bool = False,
        slim_neck_repr_compressor: bool = True,
        is_deconvolutional_decoder: bool = True,
        is_cifar_decoder: bool = True,
        binarize_repr: bool = False,
        input_preprocessor: Optional[thnn.Module] = None,
        differentiable_inference: bool = False,
        sum_of_softmaxes_inference: bool = False,
        suppress_stochastic_inference: bool = False,
        output_logits: bool = False,
        headless_mode: bool = False,
    ) -> None:
        super().__init__()

        # ----------------------------------------------------------------------
        # Check validity of arguments and auto-set values
        if input_data_height < 0 or input_data_width < 0 or input_data_channels < 0:
            raise ValueError("Input data sizes must be a positive integers!")
        input_data_size: int = (
            input_data_height * input_data_width * input_data_channels
        )

        if wrapped_repr_size < 0:
            raise ValueError(
                "Wrapped model representation size must be a positive integer!"
            )
        if compressed_input_data_size < 0:
            raise ValueError("Compressed input data size must be a positive integer!")
        if compressed_repr_data_size < 0:
            raise ValueError(
                "Compressed representation data size must be a positive integer!"
            )
        if shared_musigma_layer_size < 0:
            raise ValueError("Shared mu/sigma layer size must be a positive integer!")
        if sampled_code_size < 0:
            raise ValueError("Sampled code size must be a positive integer!")
        if ensemble_numerosity < 0 and not suppress_stochastic_inference:
            raise ValueError("Ensemble numerosity must be a positive integer!")

        # Check compatibility of arguments
        if input_data_no_compress and input_data_conv_flatten:
            raise ValueError(
                "Only one of input_data_no_compress and input_data_conv_flatten can be True!"
            )

        # Raise warnings and auto-set values
        if input_data_no_compress:
            warnings.warn(
                "If input_data_no_compress is True, compressed_input_data_size and convolutional_input_compressor values are ignored!",
                UserWarning,
            )
            compressed_input_data_size = input_data_size
            convolutional_input_compressor = False

        if input_data_conv_flatten:
            warnings.warn(
                "If input_data_conv_flatten is True, compressed_input_data_size and convolutional_input_compressor values are ignored!",
                UserWarning,
            )
            convolutional_input_compressor = True

        if repr_data_no_compress:
            warnings.warn(
                "If repr_data_no_compress is True, compressed_repr_data_size and small_neck_repr_compressor values are ignored!",
                UserWarning,
            )
            compressed_repr_data_size = wrapped_repr_size
            slim_neck_repr_compressor = False

        if suppress_stochastic_inference:
            warnings.warn(
                "If suppress_stochastic_inference is True, ensemble_numerosity value is ignored!",
                UserWarning,
            )
            ensemble_numerosity = 1
        # ----------------------------------------------------------------------

        # Wrapped model(s)
        self.wrapped_model: thnn.Module = copy.deepcopy(wrapped_model)
        if differentiable_inference:
            self.differentiable_inference: bool = True
            self.wrapped_model_clone: thnn.Module = copy.deepcopy(wrapped_model)
        else:
            self.differentiable_inference: bool = False
        # ----------------------------------------------------------------------

        # Subnetworks
        if input_data_no_compress:
            self.input_compressor: thnn.Module = thnn.Flatten()
        elif input_data_conv_flatten:
            self.input_compressor: thnn.Module = ebthnn.ConvolutionalFlattenLayer(
                input_data_height,
                input_data_width,
                detail_size=4,
                channels_in=input_data_channels,
                bias=True,
                actually_flatten=True,
            )
            compressed_input_data_size = self.input_compressor.output_numel()
        elif convolutional_input_compressor:
            self.input_compressor: thnn.Module = cnn_compressor_dispatcher_flatout(
                input_data_channels, 128
            )
            raise NotImplementedError(
                "compressed_input_data_size computation not yet implemented (and never will, probably)!"
            )
        else:
            self.input_compressor: thnn.Module = fcn_compressor_dispatcher(
                input_data_size, compressed_input_data_size, slim_neck=False
            )

        if repr_data_no_compress:
            _repr_compressor: thnn.Module = thnn.Flatten()
        elif slim_neck_repr_compressor:
            _repr_compressor: thnn.Module = fcn_compressor_dispatcher(
                wrapped_repr_size, compressed_repr_data_size, slim_neck=True
            )
        else:
            _repr_compressor: thnn.Module = fcn_compressor_dispatcher(
                wrapped_repr_size, compressed_repr_data_size, slim_neck=False
            )

        if binarize_repr:
            compressed_repr_process_fx = ebthnn.BinarizeLayer(threshold=0.5)
        else:
            compressed_repr_process_fx = thnn.Identity()

        self.repr_compressor: thnn.Module = thnn.Sequential(
            _repr_compressor, compressed_repr_process_fx
        )

        self.enc_neck: thnn.Module
        self.enc_mu: thnn.Module
        self.enc_sigma: thnn.Module
        self.dec: thnn.Module
        self.enc_neck, self.enc_mu, self.enc_sigma, self.dec = encdec_dispatcher(
            compressed_input_data_size,
            compressed_repr_data_size,
            shared_musigma_layer_size,
            sampled_code_size,
            output_size=input_data_size,
            input_channels=compressed_repr_data_size + sampled_code_size,
            deconvolutional=is_deconvolutional_decoder,
            cifar=is_cifar_decoder,
        )
        self.grps: thnn.Module = ebthnn.GaussianReparameterizerSampler()

        self.wrapped_model.train(False)
        self.wrapped_model.zero_grad(set_to_none=True)
        if differentiable_inference:
            self.wrapped_model_clone.train(False)
            self.wrapped_model_clone.zero_grad(set_to_none=True)

        self.sampled_code_size: int = sampled_code_size
        self.ensemble_numerosity: int = ensemble_numerosity
        self.sum_of_softmaxes_inference: bool = sum_of_softmaxes_inference
        self.suppress_stochastic_inference: bool = suppress_stochastic_inference
        self.deconvolutional_decoder: bool = is_deconvolutional_decoder
        self.output_logits: bool = output_logits
        self.headless_mode: bool = headless_mode

        self.notify_train_eval_changes_is_armed: bool = False
        self.notify_train_eval_changes_is_hardened: bool = False
        self.repr_layers_names_lookup: Optional[Tuple[str, ...]] = None

        self.data_shape_unbatched: Tuple[int, int, int] = (
            input_data_channels,
            input_data_height,
            input_data_width,
        )

        if self.sum_of_softmaxes_inference:
            self.process_softmax: Callable[
                [th.Tensor], th.Tensor
            ] = carso_softmax_identity
        else:
            self.process_softmax: Callable[
                [th.Tensor], th.Tensor
            ] = carso_softmax_harden_max

        self.input_preprocessor: Optional[thnn.Module] = input_preprocessor
        if input_preprocessor is not None:
            self.input_preprocessor: Optional[thnn.Module] = copy.deepcopy(
                input_preprocessor
            )
            self.input_preprocessor.train(False)
            self.input_preprocessor.zero_grad(set_to_none=True)

    # --------------------------------------------------------------------------

    def train(self: T, mode: bool = True) -> T:
        if self.notify_train_eval_changes_is_armed and self.training != mode:
            if self.notify_train_eval_changes_is_hardened:
                raise RuntimeError(
                    f"Change of training mode from {self.training} to {mode} detected, but denied since the model has been hardened."
                )
            warnings.warn(
                f"Change of training mode from {self.training} to {mode} detected. Allowing it.",
                UserWarning,
            )

        if not isinstance(mode, bool):
            raise ValueError("Training mode is expected to be boolean")
        self.training: bool = mode
        for module in self.children():
            module.train(mode)

        self.wrapped_model.train(False)
        if self.input_preprocessor is not None:
            self.input_preprocessor.train(False)
        if self.differentiable_inference:
            self.wrapped_model_clone.train(False)

        if mode:
            self.wrapped_model.zero_grad(set_to_none=True)
            ebthnn.utils.model_reqgrad_(self.wrapped_model, False)
            if self.input_preprocessor is not None:
                self.input_preprocessor.zero_grad(set_to_none=True)
            if self.differentiable_inference:
                self.wrapped_model_clone.zero_grad(set_to_none=True)
        else:
            if not self.differentiable_inference:
                self.wrapped_model.zero_grad(set_to_none=True)
                ebthnn.utils.model_reqgrad_(self.wrapped_model, False)
                if self.input_preprocessor is not None:
                    self.input_preprocessor.zero_grad(set_to_none=True)
                    ebthnn.utils.model_reqgrad_(self.input_preprocessor, False)
            else:
                self.wrapped_model.zero_grad(set_to_none=False)
                self.wrapped_model_clone.zero_grad(set_to_none=False)
                ebthnn.utils.model_reqgrad_(self.wrapped_model, True)
                if self.input_preprocessor is not None:
                    self.input_preprocessor.zero_grad(set_to_none=False)
                    ebthnn.utils.model_reqgrad_(self.input_preprocessor, True)

        return self

    # --------------------------------------------------------------------------

    def parameters_to_train(self: T, *args, **kwargs) -> Iterable[th.Tensor]:
        return itertools.chain(
            self.input_compressor.parameters(*args, **kwargs),
            self.repr_compressor.parameters(*args, **kwargs),
            self.enc_neck.parameters(*args, **kwargs),
            self.enc_mu.parameters(*args, **kwargs),
            self.enc_sigma.parameters(*args, **kwargs),
            self.dec.parameters(*args, **kwargs),
            self.grps.parameters(*args, **kwargs),
        )

    def parameters(self: T, *args, **kwargs) -> Iterable[th.Tensor]:
        return self.parameters_to_train(*args, **kwargs)

    def set_repr_layers_names_lookup(
        self: T, repr_layers_names_lookup: Optional[Tuple[str, ...]]
    ) -> T:
        self.repr_layers_names_lookup: Optional[
            Tuple[str, ...]
        ] = repr_layers_names_lookup
        return self

    def notify_train_eval_changes(
        self: T, armed: bool = True, hardened: bool = False
    ) -> T:
        self.notify_train_eval_changes_is_armed: bool = armed
        self.notify_train_eval_changes_is_hardened: bool = hardened
        return self

    def get_head_if_headless(
        self: T,
        x_input_if_headless: th.Tensor,
        repr_layers_names: Optional[Tuple[str, ...]] = None,
    ) -> th.Tensor:
        if self.headless_mode:
            if repr_layers_names is None and self.repr_layers_names_lookup is not None:
                repr_layers_names = self.repr_layers_names_lookup
            elif repr_layers_names is None and self.repr_layers_names_lookup is None:
                raise ValueError(
                    "repr_layers_names must be provided if repr_layers_names_lookup is not set."
                )

            extracted_repr: th.Tensor
            _, extracted_repr, _ = ebthnn.utils.gather_model_repr(
                self.wrapped_model,
                x_input_if_headless,
                repr_layers_names,
                preserve_graph=self.differentiable_inference,
            )
            del _
            return extracted_repr
        else:
            raise RuntimeError(
                "The get_head_if_headless method should only be used in headless mode."
            )

    # --------------------------------------------------------------------------

    def forward(
        self: T, x_input: th.Tensor, repr_layers_names: Optional[Tuple[str, ...]] = None
    ) -> Union[Tuple[th.Tensor, Tuple[th.Tensor, th.Tensor]], th.Tensor]:
        if repr_layers_names is None and self.repr_layers_names_lookup is not None:
            repr_layers_names = self.repr_layers_names_lookup
        elif repr_layers_names is None and self.repr_layers_names_lookup is None:
            raise ValueError(
                "repr_layers_names must be provided if repr_layers_names_lookup is not set."
            )

        if self.training:
            with th.no_grad():
                x_input.requires_grad_(False)
                extracted_repr: th.Tensor
                _, extracted_repr, _ = ebthnn.utils.gather_model_repr(
                    self.wrapped_model, x_input, repr_layers_names, preserve_graph=False
                )
                del _
            x_input.requires_grad_(True)
            extracted_repr.requires_grad_(True)
            if self.input_preprocessor is not None:
                x_input: th.Tensor = self.input_preprocessor(x_input)
            compressed_input: th.Tensor = self.input_compressor(x_input)
            compressed_repr: th.Tensor = self.repr_compressor(extracted_repr)
            cvae_encoder_input: th.Tensor = th.cat(
                (compressed_input, compressed_repr), dim=1
            )
            cvae_precompressed: th.Tensor = self.enc_neck(cvae_encoder_input)

            cvae_mu: th.Tensor
            cvae_sigma: th.Tensor
            cvae_mu, cvae_sigma = self.enc_mu(cvae_precompressed), self.enc_sigma(
                cvae_precompressed
            )
            dist_params: Tuple[th.Tensor, th.Tensor] = (cvae_mu, cvae_sigma)
            cvae_encoded: th.Tensor = self.grps(cvae_mu, cvae_sigma)
            cvae_decoder_input: th.Tensor = th.cat(
                (cvae_encoded, compressed_repr), dim=1
            )

            return self.dec(cvae_decoder_input).reshape(*x_input.shape), dist_params

        else:
            with contextlib.ExitStack() as stack:
                if self.differentiable_inference:
                    stack.enter_context(th.enable_grad())
                    x_input.requires_grad_(True)
                else:
                    stack.enter_context(th.no_grad())
                    x_input: th.Tensor = x_input.detach().clone()

                if not self.headless_mode:
                    extracted_repr: th.Tensor
                    _, extracted_repr, _ = ebthnn.utils.gather_model_repr(
                        self.wrapped_model,
                        x_input,
                        repr_layers_names,
                        preserve_graph=self.differentiable_inference,
                    )
                    del _
                else:
                    extracted_repr: th.Tensor = x_input

                compressed_repr: th.Tensor = self.repr_compressor(extracted_repr)

                if self.differentiable_inference:
                    ensembled_classifier: thnn.Module = self.wrapped_model_clone
                else:
                    ensembled_classifier: thnn.Module = self.wrapped_model

                ret_out: th.Tensor = th.exp(
                    carso_ensembled_classifier_ensemblebatched(
                        ensembled_classifier,
                        self.dec,
                        compressed_repr.repeat(self.ensemble_numerosity, 1).reshape(
                            self.ensemble_numerosity, *compressed_repr.shape
                        ),
                        compressed_repr.shape[0],
                        self.sampled_code_size,
                        x_input.shape
                        if not self.headless_mode
                        else (
                            extracted_repr.shape[0],
                            *self.data_shape_unbatched,
                        ),
                        self.deconvolutional_decoder,
                    )
                )
                ret_out: th.Tensor = self.process_softmax(ret_out).sum(dim=0)
                ret_out: th.Tensor = th.softmax(ret_out, dim=1)

                if self.output_logits:
                    return th.log(ret_out / (1 - ret_out))
                else:
                    return ret_out
