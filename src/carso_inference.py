#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---- IMPORTS ----
import argparse

import torch as th
from inference.tooling.carso_infer import carso_ensembled_infer
from training.tooling.architectures import compressor_dispatcher
from training.tooling.architectures import fcn_carso_dispatcher
from training.tooling.architectures import mnistfcn_dispatcher
from training.tooling.attacks import attacks_dispatcher
from training.tooling.data import mnist_dataloader_dispatcher


def attack_shorthand(model_arg, shorthand):
    attack_library = attacks_dispatcher(model_arg, True, True, True, True, True, True)
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


def main():
    # Argument parsing...
    parser = argparse.ArgumentParser(
        description="FCN+CARSO on MNIST inference and comparison"
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=1500,
        metavar="N",
        help="Number of voters in the ensemble (default: 1500)",
    )
    parser.add_argument(
        "--attack",
        type=str,
        default="dfl",
        metavar="atk",
        help="Shorthand name of the attack to use (one of: 'pgd', 'fgs', 'dfl')",
    )
    parser.add_argument(
        "--strong",
        action="store_true",
        default=False,
        help="Attacks the model with a strong(er) epsilon",
    )
    parser.add_argument(
        "--strongest",
        action="store_true",
        default=False,
        help="Attacks the model with the strongest epsilon; overrides all other strength specifications",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=False,
        help="Disables CUDA acceleration during training",
    )
    parser.add_argument(
        "--kwta",
        action="store_true",
        default=False,
        help="Constrain neuron activation with kWTA selection",
    )
    args = parser.parse_args()

    # Inference constants
    TEST_BATCHSIZE: int = 512

    # ---- DEVICE HANDLING ----
    use_cuda = not args.no_cuda and th.cuda.is_available()
    device = th.device("cuda" if use_cuda else "cpu")

    # ---- DATASETS ----
    _, test_dl, _ = mnist_dataloader_dispatcher(
        batch_size_train=256,
        batch_size_test=TEST_BATCHSIZE,
        cuda_accel=bool(device == th.device("cuda")),
    )
    del _

    # ---- MODEL DEFINITION / INSTANTIATION ----
    vanilla_classifier = mnistfcn_dispatcher(device=device, kwta_filter=bool(args.kwta))
    adversarial_classifier = mnistfcn_dispatcher(
        device=device, kwta_filter=bool(args.kwta)
    )
    repr_funnel = compressor_dispatcher(290, 290 // 5, device=device)
    _, _, _, carso_decoder = fcn_carso_dispatcher(
        28 * 28 // 4,
        290 // 5,
        (28 * 28 // 4 + 290 // 5 + 36) // 2,
        36,
        28 * 28,
        device=device,
    )
    del _

    # Load pre-trained models
    vanilla_classifier.load_state_dict(th.load("../models/mnist_fcn_clean.pth"))
    adversarial_classifier.load_state_dict(th.load("../models/mnist_fcn_adv.pth"))
    repr_funnel.load_state_dict(th.load("../models/repr_funnel_adv.pth"))
    carso_decoder.load_state_dict(th.load("../models/carso_dec_adv.pth"))

    # Move models to appropriate device and put them in evaluation mode
    vanilla_classifier = vanilla_classifier.eval()
    adversarial_classifier = adversarial_classifier.eval()
    repr_funnel = repr_funnel.eval()
    carso_decoder = carso_decoder.eval()
    vanilla_classifier.to(device)
    adversarial_classifier.to(device)
    repr_funnel.to(device)
    carso_decoder.to(device)

    # Determine attacks
    if args.strong:
        strength = "s"
    else:
        strength = "w"
    if args.strongest:
        strength = "x"
    attack_clean = attack_shorthand(vanilla_classifier, args.attack + strength)
    attack_adv = attack_shorthand(adversarial_classifier, args.attack + strength)

    # ---- RESULT STORE ----
    NUMBER_OF_ELEM = 0
    VANILLA_CORRECT = 0
    VANILLA_ADV_CORRECT = 0
    ADVERSARIAL_CLEAN_CORRECT = 0
    ADVERSARIAL_ATTACKED_CORRECT = 0
    CARSO_CORRECT = 0
    CARSO_ADV_CORRECT = 0

    # ---- TESTING LOOP ----
    for _, batched_datapoint in enumerate(test_dl):
        true_data, true_label = batched_datapoint
        true_data, true_label = true_data.to(device), true_label.to(device)
        fake_data = attack_clean.perturb(true_data.flatten(start_dim=1), true_label)
        fake_data_adv = attack_adv.perturb(true_data.flatten(start_dim=1), true_label)

        # Just to be sure...
        vanilla_classifier = vanilla_classifier.eval()
        adversarial_classifier = adversarial_classifier.eval()
        repr_funnel = repr_funnel.eval()
        carso_decoder = carso_decoder.eval()

        with th.no_grad():
            # Classify with vanilla classifier
            vanilla_clean_class = vanilla_classifier(true_data).argmax(
                dim=1, keepdim=True
            )
            vanilla_pertu_class = vanilla_classifier(fake_data).argmax(
                dim=1, keepdim=True
            )
            # Classify with the adversarial classifier
            adversarial_clean_class = adversarial_classifier(true_data).argmax(
                dim=1, keepdim=True
            )
            adversarial_pertu_class = adversarial_classifier(fake_data_adv).argmax(
                dim=1, keepdim=True
            )
            # Classify with CARSO
            repr_layers = (
                "2.module_battery.1",
                "2.module_battery.5",
                "2.module_battery.9",
            )
            carso_clean_class = carso_ensembled_infer(
                adversarial_classifier,
                repr_funnel,
                carso_decoder,
                true_data,
                36,
                list(repr_layers),
                args.ensemble_size,
                device,
            )
            carso_pertu_class = carso_ensembled_infer(
                adversarial_classifier,
                repr_funnel,
                carso_decoder,
                fake_data_adv,
                36,
                list(repr_layers),
                args.ensemble_size,
                device,
            )

            # Prepare true / vanilla classes for comparison
            trueclass = true_label.flatten().cpu()
            vanillaclass = vanilla_clean_class.flatten().cpu()
            adversarialclass = adversarial_clean_class.flatten().cpu()
            carsoclass = carso_clean_class.flatten().cpu()
            vanillaadv = vanilla_pertu_class.flatten().cpu()
            adversarialadv = adversarial_pertu_class.flatten().cpu()
            carsoadv = carso_pertu_class.flatten().cpu()

            # Record results
            NUMBER_OF_ELEM += true_data.shape[0]
            VANILLA_CORRECT += th.eq(trueclass, vanillaclass).count_nonzero().item()
            ADVERSARIAL_CLEAN_CORRECT += (
                th.eq(trueclass, adversarialclass).count_nonzero().item()
            )
            CARSO_CORRECT += th.eq(trueclass, carsoclass).count_nonzero().item()
            VANILLA_ADV_CORRECT += th.eq(trueclass, vanillaadv).count_nonzero().item()
            ADVERSARIAL_ATTACKED_CORRECT += (
                th.eq(trueclass, adversarialadv).count_nonzero().item()
            )
            CARSO_ADV_CORRECT += th.eq(trueclass, carsoadv).count_nonzero().item()

    # Compute accuracies
    vanilla_acc = VANILLA_CORRECT / NUMBER_OF_ELEM
    adv_acc = ADVERSARIAL_CLEAN_CORRECT / NUMBER_OF_ELEM
    carso_acc = CARSO_CORRECT / NUMBER_OF_ELEM
    van_adv_acc = VANILLA_ADV_CORRECT / NUMBER_OF_ELEM
    adv_adv_acc = ADVERSARIAL_ATTACKED_CORRECT / NUMBER_OF_ELEM
    carso_adv_acc = CARSO_ADV_CORRECT / NUMBER_OF_ELEM

    # Printout
    print("\n\n")
    print(f"VANILLA ACCURACY                  : {vanilla_acc}")
    print(f"ADVERSARIAL ACCURACY              : {adv_acc}")
    print(f"CARSO ACCURACY                    : {carso_acc}")
    print("\n")
    print(f"VANILLA ACCURACY UNDER ATTACK     : {van_adv_acc}")
    print(f"ADVERSARIAL ACCURACY UNDER ATTACK : {adv_adv_acc}")
    print(f"CARSO ACCURACY UNDER ATTACK       : {carso_adv_acc}")
    print("\n\n")


# Run!
if __name__ == "__main__":
    main()
