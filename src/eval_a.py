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
from ebtorch.data import data_prep_dispatcher_1ch
from ebtorch.data import fashionmnist_dataloader_dispatcher
from ebtorch.data import mnist_dataloader_dispatcher
from tooling.architectures import fashionmnist_cnn_classifier_dispatcher
from tooling.architectures import mnist_cnn_classifier_dispatcher
from tooling.architectures import mnist_fcn_classifier_dispatcher
from tqdm.auto import tqdm

# ------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FCN/CNN+CARSO on MNIST/FashionMNIST inference and comparison"
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
        default=0.3,
        metavar="<epsilon>",
        help="Strength of the attack (default: 0.3)",
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
    parser.add_argument(
        "--base_model_type",
        type=str,
        default="fcn",
        metavar="<base_model_type>",
        help="Type of model to use (either: fcn, cnn; default: fcn)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        metavar="<dataset>",
        help="Dataset to use (either: mnist, fashionmnist; default: mnist)",
    )
    args = parser.parse_args()

    # --------------------------------------------------------------------------

    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    batchsize_adaptation_ratio = 14

    # Dataset selection
    if args.dataset == "mnist":
        data_dispatcher = mnist_dataloader_dispatcher
    elif args.dataset == "fashionmnist":
        data_dispatcher = fashionmnist_dataloader_dispatcher
    else:
        raise ValueError("Invalid dataset selected! Valid options: mnist, fashionmnist")

    _, test_dl, _ = data_dispatcher(
        batch_size_train=1,
        batch_size_test=args.batchsize
        if not (args.e2e or args.noextract)
        else args.batchsize // batchsize_adaptation_ratio,
        cuda_accel=bool(device == th.device("cuda")),
    )
    del _
    # --------------------------------------------------------------------------

    # Model selection
    if args.base_model_type == "fcn":
        adversarial_classifier = mnist_fcn_classifier_dispatcher(kwta_filter=False)
    elif args.base_model_type == "cnn":
        if args.dataset == "mnist":
            adversarial_classifier = mnist_cnn_classifier_dispatcher()
        elif args.dataset == "fashionmnist":
            adversarial_classifier = fashionmnist_cnn_classifier_dispatcher()
        else:
            raise ValueError(
                "Invalid dataset selected! Valid options: mnist, fashionmnist"
            )
    else:
        raise ValueError("Invalid model selected! Valid options: fcn, cnn")

    adversarial_classifier.load_state_dict(
        th.load(f"../models/{args.base_model_type}_{args.dataset}_adv.pth")
    )
    adversarial_classifier.to(device).eval()

    carso_machinery = CARSOWrap(
        # Relevant
        wrapped_model=adversarial_classifier,
        input_data_height=28,
        input_data_width=28,
        input_data_channels=1,
        wrapped_repr_size=290 if args.base_model_type == "fcn" else 4814,
        compressed_repr_data_size=130,
        shared_musigma_layer_size=96,
        sampled_code_size=64,
        ensemble_numerosity=args.ensemble_numerosity,
        input_data_no_compress=False,
        input_data_conv_flatten=True,
        repr_data_no_compress=False,
        slim_neck_repr_compressor=True,
        is_deconvolutional_decoder=True,
        is_cifar_decoder=False,
        binarize_repr=False,
        input_preprocessor=(
            data_prep_dispatcher_1ch(device=device, post_flatten=False)
            if args.dataset == "mnist"
            else th.nn.Identity()
        ),
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
        th.load(
            f"../models/carso_reprcompressor_{args.base_model_type}_{args.dataset}_adv.pth"
        )
    )
    carso_machinery.dec.load_state_dict(
        th.load(f"../models/carso_dec_{args.base_model_type}_{args.dataset}_adv.pth")
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

    # Representation layers selection
    if args.base_model_type == "fcn":
        repr_layers = (
            "2.module_battery.1",
            "2.module_battery.5",
            "2.module_battery.9",
        )
    elif args.base_model_type == "cnn":
        repr_layers = ("1", "4", "9", "13")
    else:
        raise ValueError("Invalid model selected! Valid options: fcn, cnn")

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
