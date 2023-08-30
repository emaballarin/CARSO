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
from ebtorch.data import cifarten_dataloader_dispatcher
from ebtorch.data import data_prep_dispatcher_3ch
from tooling.architectures import PreActResNet18Cifar10
from tooling.attacks import attacks_dispatcher
from tqdm.auto import tqdm

# ------------------------------------------------------------------------------


def attack_shorthand(model, shorthand):
    attack_library = attacks_dispatcher(
        model,
        True,  # FGSM
        True,  # PGD
        True,  # DeepFool
        True,  # APDG CE
        False,
        True,  # Weak
        True,  # Strong
        True,  # Strongest
        dataset="cifarx",
        apgd_stochastic=False,
    )
    if shorthand == "pgdw":
        return attack_library[0]
    if shorthand == "pgds":
        return attack_library[1]
    if shorthand == "pgdx":
        return attack_library[2]
    if shorthand == "fgsw":
        return attack_library[3]
    if shorthand == "fgss":
        return attack_library[4]
    if shorthand == "fgsx":
        return attack_library[5]
    if shorthand == "dflw":
        return attack_library[6]
    if shorthand == "dfls":
        return attack_library[7]
    if shorthand == "dflx":
        return attack_library[8]
    if shorthand == "apgw":
        return attack_library[9]
    if shorthand == "apgs":
        return attack_library[10]
    if shorthand == "apgx":
        return attack_library[11]


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
        "--attack",
        type=str,
        default="pgd",
        metavar="<attack>",
        help="Type of the attack (either: pgd, fgs, dfl, apg; default: pgd)",
    )
    parser.add_argument(
        "--strength",
        type=str,
        default="s",
        metavar="<strength>",
        help="Strength of the attack (either: w, s, x; default: s)",
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
        if not args.e2e
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
        is_cifar_decoder=10,
        binarize_repr=False,
        input_preprocessor=data_prep_dispatcher_3ch(device, post_flatten=False),
        differentiable_inference=False if not args.e2e else True,
        sum_of_softmaxes_inference=False if not args.e2e else True,
        suppress_stochastic_inference=True if args.ensemble_numerosity == 1 else False,
        output_logits=False if not args.e2e else True,
        headless_mode=False,
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

    if args.e2e:
        attack_adv_model = attack_shorthand(
            model=carso_machinery, shorthand=str(args.attack) + str(args.strength)
        )
    else:
        attack_adv_model = attack_shorthand(
            model=adversarial_classifier,
            shorthand=str(args.attack) + str(args.strength),
        )

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

    for _, (true_data, true_label) in tqdm(  # type: ignore
        iterable=enumerate(test_dl), total=len(test_dl), desc="Testing batch"
    ):
        true_data, true_label = true_data.to(device), true_label.to(device)

        if args.e2e:
            carso_machinery.set_repr_layers_names_lookup(repr_layers)

        fake_data_adv = attack_adv_model.perturb(true_data, true_label)

        if args.e2e:
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
            adversarial_pertu_class = adversarial_classifier(fake_data_adv).argmax(
                dim=1, keepdim=True
            )

            # ------------------------------------------------------------------
            carso_clean_class = carso_machinery(true_data, repr_layers)
            carso_pertu_class = carso_machinery(fake_data_adv, repr_layers)
            carso_clean_class, carso_pertu_class = carso_clean_class.argmax(
                dim=1, keepdim=True
            ).to(device), carso_pertu_class.argmax(dim=1, keepdim=True).to(device)
            # ------------------------------------------------------------------

            trueclass = true_label.flatten().cpu()
            adversarialclass = adversarial_clean_class.flatten().cpu()
            carsoclass = carso_clean_class.flatten().cpu()
            adversarialadv = adversarial_pertu_class.flatten().cpu()
            carsoadv = carso_pertu_class.flatten().cpu()

            # Record results
            NUMBER_OF_ELEM += true_data.shape[0]
            ADVERSARIAL_CLEAN_CORRECT += (
                th.eq(trueclass, adversarialclass).count_nonzero().item()
            )
            CARSO_CORRECT += th.eq(trueclass, carsoclass).count_nonzero().item()
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
    adv_adv_acc = ADVERSARIAL_ATTACKED_CORRECT / NUMBER_OF_ELEM
    carso_adv_acc = CARSO_ADV_CORRECT / NUMBER_OF_ELEM
    # --------------------------------------------------------------------------

    # Printout
    print("\n\n")
    is_e2e = "E2E" if args.e2e else ""
    print(
        f"ATTACK TYPE/STRENGTH              : {str(args.attack) + str(args.strength) + is_e2e}"
    )
    print("\n")
    print(f"ADVERSARIAL ACCURACY              : {adv_acc}")
    print(f"CARSO ACCURACY                    : {carso_acc}")
    print("\n")
    print(f"ADVERSARIAL ACCURACY UNDER ATTACK : {adv_adv_acc}")
    print(f"CARSO ACCURACY UNDER ATTACK       : {carso_adv_acc}")
    print("\n\n")


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
