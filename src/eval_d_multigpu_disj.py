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
import torch.distributed as dist
from carso import CARSOWrap
from ebtorch.data import cifarten_dataloader_dispatcher
from ebtorch.data import data_prep_dispatcher_3ch
from ebtorch.distributed import slurm_nccl_env
from ebtorch.nn import WideResNet
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm


# ------------------------------------------------------------------------------
def main_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WideResNet-28-10+CARSO on CIFAR10 inference and comparison"
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
        default=800,
        metavar="<batch_size>",
        help="Batch size for testing, model-only; e2e is rescaled accordingly (default: 800)",
    )
    parser.add_argument(
        "--ensemble_numerosity",
        type=int,
        default=8,
        metavar="<batch_size>",
        help="Size of the ensemble used to perform inference (default: 6)",
    )
    args = parser.parse_args()
    return args


# ------------------------------------------------------------------------------


def main_run(args: argparse.Namespace) -> None:
    # --------------------------------------------------------------------------
    # Distributed devices setup
    (
        rank,
        world_size,
        _,
        cpus_per_task,
        local_rank,
        device,
    ) = slurm_nccl_env()

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    th.cuda.set_device(device)
    # --------------------------------------------------------------------------

    # Dataset/DataLoader
    bsar = (
        0.25
        * (1.12 if world_size <= 1 else 1.2)
        * 800
        / (max(800, args.batchsize) * max(4, args.ensemble_numerosity))
    )
    # Repeated twice just to allow gathering of dataset for DistributedSampler
    _, test_dl, _ = cifarten_dataloader_dispatcher(
        batch_size_train=1,
        batch_size_test=args.batchsize
        if not (args.e2e or args.noextract)
        else int(args.batchsize * bsar),
        cuda_accel=True,
        shuffle_test=False,
        unshuffle_train=True,
    )
    _, test_dl, _ = cifarten_dataloader_dispatcher(
        batch_size_train=1,
        batch_size_test=args.batchsize
        if not (args.e2e or args.noextract)
        else int(args.batchsize * bsar),
        cuda_accel=True,
        shuffle_test=False,
        unshuffle_train=True,
        dataloader_kwargs={
            "sampler": DistributedSampler(test_dl.dataset),
            "num_workers": cpus_per_task,
        },
    )
    del _

    # --------------------------------------------------------------------------

    adversarial_classifier = WideResNet(bn_momentum=0.01)
    adversarial_classifier.load_state_dict(
        th.load("../models/cifar10_a3_b10_t4_20m_w.pt")
    )
    adversarial_classifier.to(device).eval()

    carso_machinery = CARSOWrap(
        # Relevant
        wrapped_model=adversarial_classifier,
        input_data_height=32,
        input_data_width=32,
        input_data_channels=3,
        wrapped_repr_size=573450,
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
        input_preprocessor=data_prep_dispatcher_3ch(device, post_flatten=False),
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
        th.load("../models/carso_reprcompressor_cuiwrn2810_cifar10_adv.pth")
    )
    carso_machinery.dec.load_state_dict(
        th.load("../models/carso_dec_cuiwrn2810_cifar10_adv.pth")
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
        "layer.0.block.1.conv_1",
        "layer.1.block.0.shortcut",
        "layer.1.block.1.conv_1",
        "layer.1.block.2.conv_1",
        "layer.2.block.0.shortcut",
        "layer.2.block.1.conv_1",
        "layer.2.block.2.conv_1",
        "layer.2.block.3.conv_1",
        "logits",
    )

    # --------------------------------------------------------------------------
    # Evaluation counters
    number_of_elem_global_item: int = 0
    adversarial_clean_correct_global_item: int = 0
    carso_correct_global_item: int = 0
    carso_adv_correct_global_item: int = 0

    test_dl.sampler.set_epoch(0)  # type: ignore

    for _, (true_data, true_label) in tqdm(  # type: ignore
        iterable=enumerate(test_dl),
        total=len(test_dl),
        desc="Testing batch",
        disable=(local_rank != 0),
    ):
        true_data, true_label = true_data.to(device), true_label.to(device)

        if args.e2e or args.noextract:
            carso_machinery.set_repr_layers_names_lookup(repr_layers)

        if not args.noextract:
            fake_data_adv = attack_adv_model.run_standard_evaluation(
                true_data,
                true_label,
                bs=args.batchsize if not args.e2e else int(args.batchsize * bsar),
            )
        else:
            true_repr = carso_machinery.get_head_if_headless(true_data).detach().clone()
            fake_repr_adv = attack_adv_model.run_standard_evaluation(
                true_repr,
                true_label,
                bs=int(args.batchsize * bsar),
            )

        if args.e2e or args.noextract:
            carso_machinery.set_repr_layers_names_lookup(None)

        # ----------------------------------------------------------------------
        carso_machinery.sum_of_softmaxes_inference = False
        adversarial_classifier = adversarial_classifier.eval()
        carso_machinery.to(device).eval()
        # ----------------------------------------------------------------------
        with th.no_grad():
            # Classify with the adversarial classifier
            adversarial_clean_class = adversarial_classifier(true_data).argmax(
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

            trueclass = true_label.flatten()
            adversarialclass = adversarial_clean_class.flatten()
            carsoclass = carso_clean_class.flatten()
            carsoadv = carso_pertu_class.flatten()

            # Record results
            number_of_elem_global_item += true_data.shape[0].item()
            adversarial_clean_correct_global_item += (
                th.eq(trueclass, adversarialclass).count_nonzero().item()
            )
            carso_correct_global_item += (
                th.eq(trueclass, carsoclass).count_nonzero().item()
            )
            carso_adv_correct_global_item += (
                th.eq(trueclass, carsoadv).count_nonzero().item()
            )
        # --------------------------------------------------------------------------
        if local_rank == 0:
            adv_acc = adversarial_clean_correct_global_item / number_of_elem_global_item
            carso_acc = carso_correct_global_item / number_of_elem_global_item
            carso_adv_acc = carso_adv_correct_global_item / number_of_elem_global_item
            # ------------------------------------------------------------------

            # Printout
            print("\n\n")
            print(f"ADVERSARIAL ACCURACY              : {adv_acc}")
            print(f"CARSO ACCURACY                    : {carso_acc}")
            print("\n")
            print(f"CARSO ACCURACY UNDER ATTACK       : {carso_adv_acc}")
            print("\n\n")

    # --------------------------------------------------------------------------

    dist.destroy_process_group()


# ------------------------------------------------------------------------------


def main() -> None:
    parser_output = main_parse()
    main_run(parser_output)


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
