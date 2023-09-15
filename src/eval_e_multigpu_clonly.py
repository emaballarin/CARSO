#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  Copyright (c) 2023 Emanuele Ballarin <emanuele@ballarin.cc>
#  Released under the terms of the MIT License
#  (see: https://url.ballarin.cc/mitlicense)
#
# ------------------------------------------------------------------------------
import argparse

import torch as th
from carso import CARSOWrap
from ebtorch.data import cifarhundred_dataloader_dispatcher
from ebtorch.data import data_prep_dispatcher_3ch
from ebtorch.nn import WideResNet
from tqdm.auto import tqdm


# ------------------------------------------------------------------------------
def main_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WideResNet-28-10+CARSO on CIFAR100 clean inference"
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=800,
        metavar="<batch_size>",
        help="Batch size for testing, model-only; e2e is rescaled accordingly (default: 800)",
    )
    parser.add_argument(
        "--ensemble_numerosity",
        type=int,
        default=8,
        metavar="<batch_size>",
        help="Size of the ensemble used to perform inference (default: 8)",
    )
    args = parser.parse_args()
    return args


# ------------------------------------------------------------------------------


def main_run(args: argparse.Namespace) -> None:
    # Device
    device = th.device("cuda")

    # Dataset/DataLoader
    _, test_dl, _ = cifarhundred_dataloader_dispatcher(
        batch_size_train=1,
        batch_size_test=args.batchsize,
        cuda_accel=True,
        dataloader_kwargs={
            "num_workers": 16,
        },
    )
    del _

    # --------------------------------------------------------------------------

    adversarial_classifier = WideResNet(num_classes=100, bn_momentum=0.01)
    adversarial_classifier.load_state_dict(
        th.load("../models/cifar100_a5_b12_t4_50m_w.pt")
    )
    adversarial_classifier.to(device).eval()

    carso_machinery = CARSOWrap(
        # Relevant
        wrapped_model=adversarial_classifier,
        input_data_height=32,
        input_data_width=32,
        input_data_channels=3,
        wrapped_repr_size=286820,
        compressed_repr_data_size=3072,
        shared_musigma_layer_size=192,
        sampled_code_size=128,
        ensemble_numerosity=args.ensemble_numerosity,
        input_data_no_compress=False,
        input_data_conv_flatten=True,
        repr_data_no_compress=False,
        slim_neck_repr_compressor=True,
        is_deconvolutional_decoder=True,
        is_cifar_decoder=10,
        binarize_repr=False,
        input_preprocessor=data_prep_dispatcher_3ch(device, post_flatten=False),
        differentiable_inference=False,
        sum_of_softmaxes_inference=False,
        suppress_stochastic_inference=True if args.ensemble_numerosity == 1 else False,
        output_logits=False,
        headless_mode=False,
        # Forced/Dummy
        compressed_input_data_size=0,
        convolutional_input_compressor=False,
    )

    carso_machinery.repr_compressor.load_state_dict(
        th.load("../models/carso_reprcompressor_cuiwrn2810_cifar100_adv.pth")
    )
    carso_machinery.dec.load_state_dict(
        th.load("../models/carso_dec_cuiwrn2810_cifar100_adv.pth")
    )
    carso_machinery.to(device).eval()

    repr_layers = (
        "layer.1.block.0.conv_0",  # From: 04/09
        "layer.1.block.1.conv_1",  # From: 04/09
        "layer.2.block.0.conv_1",  # From: 04/09
        "layer.2.block.1.conv_1",  # From: ADD
        "layer.2.block.2.conv_1",  # From: 04/09
        "logits",  # From: 04/09
    )

    # --------------------------------------------------------------------------
    # Evaluation counters
    number_of_elem_global_item: int = 0
    carso_correct_global_item: int = 0

    for _, (true_data, true_label) in tqdm(  # type: ignore
        iterable=enumerate(test_dl),
        total=len(test_dl),
        desc="Testing batch",
    ):
        true_data, true_label = true_data.to(device), true_label.to(device)
        adversarial_classifier = adversarial_classifier.eval()
        carso_machinery.to(device).eval()
        # ----------------------------------------------------------------------
        with th.no_grad():
            # Inference
            carsoclass = (
                carso_machinery(true_data, repr_layers)
                .argmax(dim=1, keepdim=True)
                .to(device)
                .flatten()
            )
            trueclass = true_label.flatten()

            # Record results
            number_of_elem_global_item += true_data.shape[0]
            carso_correct_global_item += (
                th.eq(trueclass, carsoclass).count_nonzero().item()
            )
    # --------------------------------------------------------------------------
    carso_acc = carso_correct_global_item / number_of_elem_global_item
    # ------------------------------------------------------------------

    # Printout
    print("\n\n")
    print(f"CARSO ACCURACY                    : {carso_acc}")
    print("\n\n")


# ------------------------------------------------------------------------------


def main() -> None:
    parser_output = main_parse()
    main_run(parser_output)


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
