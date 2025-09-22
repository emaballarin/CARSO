#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
#  Copyright (c) 2025 Emanuele Ballarin <emanuele@ballarin.cc>
#  Released under the terms of the MIT License
#  (see: https://url.ballarin.cc/mitlicense)
# ──────────────────────────────────────────────────────────────────────────────
import argparse
from typing import Tuple

import autoattack as aatk
import torch as th
import torch.distributed as dist
from carso import CARSOWrap
from ebtorch.data import data_prep_dispatcher_3ch
from ebtorch.data import tinyimagenet_dataloader_dispatcher
from ebtorch.distributed import slurm_nccl_env
from ebtorch.nn import WideResNet
from ebtorch.nn.architectures_resnets_dm import TINY_MEAN
from ebtorch.nn.architectures_resnets_dm import TINY_STD
from safetensors.torch import load_model
from tooling.pgdeot import PGD as PGDEoT
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm


# ──────────────────────────────────────────────────────────────────────────────
BASE_MODEL_NAME: str = "wrn_28_10"
DATASET_NAME: str = "tinyimagenet_200"
MODEL_REFERENCE: str = "wang_2023"

COMPR_COND_DIM: int = 768
JOINT_LATENT_DIM: int = 192

SINGLE_GPU_WORKERS: int = 16

# noinspection DuplicatedCode
MODELS_PATH_BASE: str = "../models/"

BSAR: float = 1


# ──────────────────────────────────────────────────────────────────────────────
def main_parse() -> argparse.Namespace:
    # noinspection DuplicatedCode
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=f"{BASE_MODEL_NAME}+CARSO on {DATASET_NAME} Testing"
    )
    parser.add_argument(
        "--dist",
        action="store_true",
        default=False,
        help="Perform distributed training by means of DDP (default: False)",
    )
    parser.add_argument(
        "--e2e",
        action="store_true",
        default=True,
        help="Evaluate robustness end-to-end (default: True)",
    )
    parser.add_argument(
        "--pgdeot",
        action="store_true",
        default=False,
        help="Evaluate adversarial robustness with the PGD+EoT pipeline (default: False)",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=8 / 255,
        metavar="<epsilon>",
        help="Perturbation strength (default: 8/255)",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=80,
        metavar="<batch_size>",
        help="Batch size for evaluation (default: 80)",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=8,
        metavar="<n_samples>",
        help="Number of sampled recosntructions to classify (default: 8)",
    )
    parser.add_argument(
        "--agg",
        type=str,
        default="peel",
        metavar="<aggregation_method>",
        help="Aggregation method for model outputs (default: PeeL)",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────


def main_run(args: argparse.Namespace) -> None:
    # noinspection DuplicatedCode
    if args.dist:
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
    else:
        world_size: int = 1
        cpus_per_task: int = SINGLE_GPU_WORKERS
        local_rank: int = 0
        device: th.device = th.device("cuda" if th.cuda.is_available() else "cpu")

    # ──────────────────────────────────────────────────────────────────────────

    batchsize: int = int((args.batchsize // world_size) * (BSAR if args.e2e else 1))

    _, test_dl, _ = tinyimagenet_dataloader_dispatcher(
        batch_size_train=1,
        batch_size_test=batchsize,
        augment_train=False,
        cuda_accel=True,
        shuffle_test=not args.dist,
        unshuffle_train=args.dist,
        dataloader_kwargs=(
            {
                "num_workers": cpus_per_task,
                "persistent_workers": True,
            }
            if not args.dist
            else {}
        ),
    )
    del _

    if args.dist:
        _, test_dl, _ = tinyimagenet_dataloader_dispatcher(
            batch_size_train=1,
            batch_size_test=batchsize,
            augment_train=False,
            cuda_accel=True,
            shuffle_test=False,
            unshuffle_train=True,
            dataloader_kwargs={
                "sampler": DistributedSampler(test_dl.dataset),
                "num_workers": cpus_per_task,
                "persistent_workers": True,
            },
        )
        del _
    # ──────────────────────────────────────────────────────────────────────────
    # noinspection DuplicatedCode
    adversarial_classifier: WideResNet = WideResNet(200, bn_momentum=0.01, mean=TINY_MEAN, std=TINY_STD, autopool=True)
    load_model(adversarial_classifier, "../models/tiny_linf_wrn28_10_w.safetensors")
    adversarial_classifier.to(device).eval()

    full_repr_layers: Tuple[str, ...] = (
        "layer.0.block.0.conv_0",
        "layer.0.block.0.conv_1",
        "layer.0.block.1.conv_0",
        "layer.0.block.1.conv_1",
        "layer.0.block.2.conv_0",
        "layer.0.block.2.conv_1",
        "layer.0.block.3.conv_0",
        "layer.0.block.3.conv_1",
        "layer.1.block.0.conv_0",
        "layer.1.block.0.conv_1",
        "layer.1.block.0.shortcut",
        "layer.1.block.1.conv_0",
        "layer.1.block.1.conv_1",
        "layer.1.block.2.conv_0",
        "layer.1.block.2.conv_1",
        "layer.1.block.3.conv_0",
        "layer.1.block.3.conv_1",
        "layer.2.block.0.conv_0",
        "layer.2.block.0.conv_1",
        "layer.2.block.0.shortcut",
        "layer.2.block.1.conv_0",
        "layer.2.block.1.conv_1",
        "layer.2.block.2.conv_0",
        "layer.2.block.2.conv_1",
        "layer.2.block.3.conv_0",
        "layer.2.block.3.conv_1",
    )

    carso_machinery: CARSOWrap = CARSOWrap(
        wrapped_model=adversarial_classifier,
        input_preproc=data_prep_dispatcher_3ch(device, post_flatten=False, dataset="tinyimagenet"),
        input_shape=(3, 64, 64),
        repr_layers=full_repr_layers,
        compr_cond_dim=COMPR_COND_DIM,
        joint_latent_dim=JOINT_LATENT_DIM,
        ensemble_size=args.nsamples,
        differentiable_infer=args.e2e,
        agg_method=args.agg,
    )

    # noinspection DuplicatedCode
    _ = load_model(
        carso_machinery.repr_compressors,
        f"{MODELS_PATH_BASE}{BASE_MODEL_NAME}_{DATASET_NAME}_{MODEL_REFERENCE}_repr_compressors.safetensors",
    )
    _ = load_model(
        carso_machinery.repr_fcn_compressor,
        f"{MODELS_PATH_BASE}{BASE_MODEL_NAME}_{DATASET_NAME}_{MODEL_REFERENCE}_repr_fcn_compressor.safetensors",
    )
    _ = load_model(
        carso_machinery.decoder,
        f"{MODELS_PATH_BASE}{BASE_MODEL_NAME}_{DATASET_NAME}_{MODEL_REFERENCE}_decoder.safetensors",
    )

    carso_machinery.to(device).eval()

    # ──────────────────────────────────────────────────────────────────────────
    atk_dict_args = {
        "norm": "Linf",
        "eps": args.eps,
        "version": "rand" if args.e2e else "standard",
        "verbose": True,
    }

    # noinspection DuplicatedCode
    if args.pgdeot:
        attack_adv_model: PGDEoT = PGDEoT(carso_machinery if args.e2e else adversarial_classifier)
    else:
        attack_adv_model: aatk.AutoAttack = aatk.AutoAttack(
            carso_machinery if args.e2e else adversarial_classifier, **atk_dict_args
        )
    # ──────────────────────────────────────────────────────────────────────────
    test_dl.sampler.set_epoch(0)  # type: ignore
    n_instances: int = 0
    n_carso_correct_clean: int = 0
    n_carso_correct_adv: int = 0
    # ──────────────────────────────────────────────────────────────────────────

    for _, (true_data, true_label) in tqdm(
        enumerate(test_dl),
        total=len(test_dl),
        desc="Batch",
        disable=(local_rank != 0),
    ):
        true_data, true_label = true_data.to(device), true_label.to(device)  # type: ignore

        # ──────────────────────────────────────────────────────────────────────
        fake_data = (
            attack_adv_model(true_data, true_label)
            if args.pgdeot
            else attack_adv_model.run_standard_evaluation(
                true_data,
                true_label,
                bs=batchsize,
            )
        )
        # ──────────────────────────────────────────────────────────────────────
        adversarial_classifier.eval()
        carso_machinery.eval()
        # ──────────────────────────────────────────────────────────────────────

        with th.no_grad():
            carso_clean_class = carso_machinery(true_data).argmax(dim=1, keepdim=True).to(device).flatten()
            carso_adv_class = carso_machinery(fake_data).argmax(dim=1, keepdim=True).to(device).flatten()

            # ──────────────────────────────────────────────────────────────────
            n_instances += true_data.shape[0]
            n_carso_correct_clean += th.eq(true_label.flatten(), carso_clean_class).count_nonzero().item()
            n_carso_correct_adv += th.eq(true_label.flatten(), carso_adv_class).count_nonzero().item()
            # ──────────────────────────────────────────────────────────────────

    carso_clean_acc = n_carso_correct_clean / n_instances
    carso_adv_acc = n_carso_correct_adv / n_instances

    # ──────────────────────────────────────────────────────────────────────────

    # Printout
    print("\n\n")
    print(f"CARSO ACCURACY                    : {carso_clean_acc}")
    print("\n")
    print(f"CARSO ACCURACY UNDER ATTACK       : {carso_adv_acc}")
    print("\n\n")

    if args.dist:
        dist.destroy_process_group()


# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    main_run(main_parse())


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
# ──────────────────────────────────────────────────────────────────────────────
