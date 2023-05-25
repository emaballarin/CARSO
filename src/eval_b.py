#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  Copyright (c) 2023 Emanuele Ballarin <emanuele@ballarin.cc>
#  Released under the terms of the MIT License
#  (see: https://url.ballarin.cc/mitlicense)
#
# ------------------------------------------------------------------------------
import argparse

import autoattack as aatk
import torch as th
from carso import CARSOWrap
from tooling.architectures import cifar_data_prep_dispatcher
from tooling.architectures import PreActResNet18Cifar10
from tooling.data import cifarten_dataloader_dispatcher
from tqdm.auto import tqdm

# ------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PreActResNet18+CARSO on CIFAR10 inference and comparison"
    )
    parser.add_argument(
        "--e2e",
        action="store_true",
        default=False,
        help="Attack CARSO end-to-end, not just the wrapped model (default: False)",
    )
    parser.add_argument(
        "--noextract",
        action="store_true",
        default=False,
        help="Attack CARSO end-to-end, from the representation down only (default: False)",
    )
    parser.add_argument(
        "--explicitly_random",
        action="store_true",
        default=False,
        help="Explicitly acknowledge randomness of the defence for robustness evaluation (default: False)",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=8 / 255,
        metavar="<epsilon>",
        help="Strength of the attack (default: 8/255)",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=1536,
        metavar="<batch_size>",
        help="Batch size for testing, model-only; e2e is rescaled accordingly (default: 1536)",
    )
    parser.add_argument(
        "--ensemble_numerosity",
        type=int,
        default=4,
        metavar="<batch_size>",
        help="Size of the ensemble used to perform inference (default: 4)",
    )
    args = parser.parse_args()

    # --------------------------------------------------------------------------

    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    batchsize_adaptation_ratio = 14
    _, test_dl, _ = cifarten_dataloader_dispatcher(
        batch_size_train=1,
        batch_size_test=args.batchsize
        if not (args.e2e or args.noextract)
        else args.batchsize // batchsize_adaptation_ratio,
        cuda_accel=bool(device == th.device("cuda")),
    )
    del _

    # --------------------------------------------------------------------------

    adversarial_classifier = PreActResNet18Cifar10(device=device)
    adversarial_classifier.model.load_state_dict(
        th.load("../models/cifar_model_weights_30_epochs.pth")
    )
    adversarial_classifier.to(device).eval()

    carso_machinery = CARSOWrap(
        # Relevant
        wrapped_model=adversarial_classifier,
        input_data_height=32,
        input_data_width=32,
        input_data_channels=3,
        wrapped_repr_size=204810,
        compressed_repr_data_size=512,
        shared_musigma_layer_size=192,
        sampled_code_size=128,
        ensemble_numerosity=args.ensemble_numerosity,
        input_data_no_compress=False,
        input_data_conv_flatten=True,
        repr_data_no_compress=False,
        slim_neck_repr_compressor=True,
        is_deconvolutional_decoder=True,
        is_cifar_decoder=True,
        binarize_repr=False,
        input_preprocessor=cifar_data_prep_dispatcher(device, post_flatten=False),
        differentiable_inference=False if not (args.e2e or args.noextract) else True,
        sum_of_softmaxes_inference=False if not (args.e2e or args.noextract) else True,
        suppress_stochastic_inference=True if args.ensemble_numerosity == 1 else False,
        output_logits=False if not (args.e2e or args.noextract) else True,
        headless_mode=False if not args.noextract else True,
        # Forced/Dummy
        compressed_input_data_size=0,
        convolutional_input_compressor=False,
    )

    carso_machinery.repr_compressor.load_state_dict(
        th.load("../models/carso_reprcompressor_wongrn18_cifar10_adv.pth")
    )
    carso_machinery.dec.load_state_dict(
        th.load("../models/carso_dec_wongrn18_cifar10_adv.pth")
    )
    carso_machinery.to(device).eval()

    atk_dict_args = {
        "norm": "Linf",
        "eps": args.eps,
        "version": "rand" if args.explicitly_random else "standard",
        "verbose": False,
    }
    if args.e2e or args.noextract:
        attack_adv_model = aatk.AutoAttack(carso_machinery, **atk_dict_args)
    else:
        attack_adv_model = aatk.AutoAttack(adversarial_classifier, **atk_dict_args)

    repr_layers = (
        "model.layer1.1.conv2",
        "model.layer2.0.conv2",
        "model.layer2.0.shortcut.0",
        "model.layer3.0.conv2",
        "model.layer3.0.shortcut.0",
        "model.layer3.1.conv2",
        "model.layer4.0.conv2",
        "model.layer4.0.shortcut.0",
        "model.layer4.1.conv2",
        "model.linear",
    )

    # --------------------------------------------------------------------------

    NUMBER_OF_ELEM = 0
    ADVERSARIAL_CLEAN_CORRECT = 0
    ADVERSARIAL_ATTACKED_CORRECT = 0
    CARSO_CORRECT = 0
    CARSO_ADV_CORRECT = 0

    print("\nTesting...")

    for _, (true_data, true_label) in tqdm(
        iterable=enumerate(test_dl), total=len(test_dl), desc="Testing batch"
    ):
        true_data, true_label = true_data.to(device), true_label.to(device)

        if args.e2e or args.noextract:
            carso_machinery.set_repr_layers_names_lookup(repr_layers)

        if not args.noextract:
            fake_data_adv = attack_adv_model.run_standard_evaluation(
                true_data,
                true_label,
                bs=args.batchsize
                if not args.e2e
                else args.batchsize // batchsize_adaptation_ratio,
            )
        else:
            true_repr = carso_machinery.get_head_if_headless(true_data).detach().clone()
            fake_repr_adv = attack_adv_model.run_standard_evaluation(
                true_repr,
                true_label,
                bs=args.batchsize // batchsize_adaptation_ratio,
            )

        if args.e2e or args.noextract:
            carso_machinery.set_repr_layers_names_lookup(None)

        # ----------------------------------------------------------------------
        carso_machinery.sum_of_softmaxes_inference = False
        adversarial_classifier = adversarial_classifier.eval()
        carso_machinery.to(device).eval()
        # ----------------------------------------------------------------------
        carso_machinery.notify_train_eval_changes(armed=True, hardened=True)
        # ----------------------------------------------------------------------

        with th.no_grad():
            # Classify with the adversarial classifier
            adversarial_clean_class = adversarial_classifier(true_data).argmax(
                dim=1, keepdim=True
            )
            if not args.noextract:
                adversarial_pertu_class = adversarial_classifier(fake_data_adv).argmax(
                    dim=1, keepdim=True
                )

            # ------------------------------------------------------------------
            if not args.noextract:
                carso_clean_class = carso_machinery(true_data, repr_layers)
                carso_pertu_class = carso_machinery(fake_data_adv, repr_layers)
            else:
                carso_clean_class = carso_machinery(true_repr, repr_layers)
                carso_pertu_class = carso_machinery(fake_repr_adv, repr_layers)
            carso_clean_class, carso_pertu_class = carso_clean_class.argmax(
                dim=1, keepdim=True
            ).to(device), carso_pertu_class.argmax(dim=1, keepdim=True).to(device)
            # ------------------------------------------------------------------

            trueclass = true_label.flatten().cpu()
            adversarialclass = adversarial_clean_class.flatten().cpu()
            carsoclass = carso_clean_class.flatten().cpu()
            if not args.noextract:
                adversarialadv = adversarial_pertu_class.flatten().cpu()
            carsoadv = carso_pertu_class.flatten().cpu()

            # Record results
            NUMBER_OF_ELEM += true_data.shape[0]
            ADVERSARIAL_CLEAN_CORRECT += (
                th.eq(trueclass, adversarialclass).count_nonzero().item()
            )
            CARSO_CORRECT += th.eq(trueclass, carsoclass).count_nonzero().item()
            if not args.noextract:
                ADVERSARIAL_ATTACKED_CORRECT += (
                    th.eq(trueclass, adversarialadv).count_nonzero().item()
                )
            CARSO_ADV_CORRECT += th.eq(trueclass, carsoadv).count_nonzero().item()

        # ----------------------------------------------------------------------
        carso_machinery.notify_train_eval_changes(armed=False)
        # ----------------------------------------------------------------------

    # Compute accuracies
    adv_acc = ADVERSARIAL_CLEAN_CORRECT / NUMBER_OF_ELEM
    carso_acc = CARSO_CORRECT / NUMBER_OF_ELEM
    if not args.noextract:
        adv_adv_acc = ADVERSARIAL_ATTACKED_CORRECT / NUMBER_OF_ELEM
    carso_adv_acc = CARSO_ADV_CORRECT / NUMBER_OF_ELEM
    # --------------------------------------------------------------------------

    # Printout
    print("\n\n")
    print(f"ADVERSARIAL ACCURACY              : {adv_acc}")
    print(f"CARSO ACCURACY                    : {carso_acc}")
    print("\n")
    if not args.noextract:
        print(f"ADVERSARIAL ACCURACY UNDER ATTACK : {adv_adv_acc}")
    print(f"CARSO ACCURACY UNDER ATTACK       : {carso_adv_acc}")
    print("\n\n")


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
