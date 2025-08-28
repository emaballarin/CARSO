#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
#  Copyright (c) 2025 Emanuele Ballarin <emanuele@ballarin.cc>
#  Released under the terms of the MIT License
#  (see: https://url.ballarin.cc/mitlicense)
# ──────────────────────────────────────────────────────────────────────────────
import argparse
from typing import Tuple

import torch as th
import torch.distributed as dist
import wandb
from carso import CARSOWrap
from ebtorch.data import data_prep_dispatcher_3ch
from ebtorch.data import tinyimagenet_dataloader_dispatcher
from ebtorch.distributed import slurm_nccl_env
from ebtorch.nn import beta_reco_bce_splitout
from ebtorch.nn import WideResNet
from ebtorch.nn.architectures_resnets_dm import TINY_MEAN
from ebtorch.nn.architectures_resnets_dm import TINY_STD
from ebtorch.nn.utils import AdverApply
from ebtorch.optim import Lookahead
from ebtorch.optim import make_beta_scheduler
from ebtorch.optim import ralah_optim
from ebtorch.optim import warmed_up_linneal
from safetensors.torch import load_model
from safetensors.torch import save_model
from tooling.attacks import attacks_dispatcher
from torch import nn
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from tqdm.auto import trange

# ──────────────────────────────────────────────────────────────────────────────
BASE_MODEL_NAME: str = "wrn_28_10"
DATASET_NAME: str = "tinyimagenet_200"
MODEL_REFERENCE: str = "wang_2023"
ATTACKS_DATASET: str = "tinyimagenetnorm"

COMPR_COND_DIM: int = 768
JOINT_LATENT_DIM: int = 192

SINGLE_GPU_WORKERS: int = 16

# noinspection DuplicatedCode
LAH_STEPS: int = 6

INIT_LR: float = 5e-9
BASE_LR_MULT: float = 1e-4
FINAL_LR_MULT: float = 1.25e-8

LR_RATIO_UP: float = 0.125
LR_RATIO_FLAT: float = 0.125

VAE_BETA_LAG_RATIO: float = LR_RATIO_UP
VAE_BETA_WARMUP_RATIO: float = LR_RATIO_FLAT / 3
VAE_BETA: float = 1.0

MODELS_PATH_BASE: str = "../models/"


# ──────────────────────────────────────────────────────────────────────────────
def main_parse() -> argparse.Namespace:
    # noinspection DuplicatedCode
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=f"{BASE_MODEL_NAME}+CARSO on {DATASET_NAME} Training"
    )
    parser.add_argument(
        "--dist",
        action="store_true",
        default=False,
        help="Perform distributed training by means of DDP (default: False)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=False,
        help="Save model weights after training (default: False)",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="Log selected metdata to Weights & Biases (default: False)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=250,
        metavar="<epochs>",
        help="Number of epochs to train for (default: 250)",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=1024,
        metavar="<batch_size>",
        help="Batch size for training (default: 1024)",
    )
    parser.add_argument(
        "--advfrac",
        type=float,
        default=0.01,
        metavar="<adversarial_fraction>",
        help="Fraction of each batch to be adversarially perturbed (default: 0.04)",
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

    batchsize: int = max(args.batchsize // world_size, 2)

    train_dl, _, _ = tinyimagenet_dataloader_dispatcher(
        batch_size_train=batchsize,
        batch_size_test=1,
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
        train_dl, _, _ = tinyimagenet_dataloader_dispatcher(
            batch_size_train=batchsize,
            batch_size_test=1,
            augment_train=False,
            cuda_accel=True,
            shuffle_test=False,
            unshuffle_train=True,
            dataloader_kwargs={
                "sampler": DistributedSampler(train_dl.dataset),
                "num_workers": cpus_per_task,
                "persistent_workers": True,
            },
        )
        del _
    # ──────────────────────────────────────────────────────────────────────────
    # noinspection DuplicatedCode
    vanilla_classifier: WideResNet = WideResNet(200, bn_momentum=0.01, mean=TINY_MEAN, std=TINY_STD, autopool=True)
    load_model(vanilla_classifier, "../models/tiny_linf_wrn28_10_w.safetensors")
    vanilla_classifier.to(device).eval()

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
        wrapped_model=vanilla_classifier,
        input_preproc=data_prep_dispatcher_3ch(device, post_flatten=False, dataset="tinyimagenet"),
        input_shape=(3, 64, 64),
        repr_layers=full_repr_layers,
        compr_cond_dim=COMPR_COND_DIM,
        joint_latent_dim=JOINT_LATENT_DIM,
        ensemble_size=0,
        differentiable_infer=False,
    )

    # noinspection DuplicatedCode
    if args.dist:
        carso_machinery: CARSOWrap = nn.SyncBatchNorm.convert_sync_batchnorm(carso_machinery)
    carso_machinery.to(device)
    if args.dist:
        carso_machinery: DDP = DDP(
            carso_machinery,
            device_ids=[local_rank],
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )
    carso_machinery.train()

    # ──────────────────────────────────────────────────────────────────────────
    optimizer: Lookahead = ralah_optim(
        carso_machinery.parameters(),
        radam_lr=(complbr := BASE_LR_MULT * batchsize),  # NOSONAR
        la_steps=LAH_STEPS,
    )

    optimizer, scheduler = warmed_up_linneal(
        optim=optimizer,
        init_lr=INIT_LR,
        steady_lr=complbr,
        final_lr=(compflr := FINAL_LR_MULT * batchsize),  # NOSONAR
        warmup_epochs=(eup := int(LR_RATIO_UP * args.epochs)),  # NOSONAR
        steady_epochs=(efl := int(LR_RATIO_FLAT * args.epochs)),
        anneal_epochs=(ean := int(args.epochs - eup - efl)),
    )

    beta_scheduler = make_beta_scheduler(VAE_BETA, VAE_BETA_LAG_RATIO, VAE_BETA_WARMUP_RATIO)

    # ──────────────────────────────────────────────────────────────────────────
    adversarial_apply = AdverApply(adversaries=attacks_dispatcher(model=vanilla_classifier, dataset=ATTACKS_DATASET))
    # ──────────────────────────────────────────────────────────────────────────
    if args.wandb and local_rank == 0:
        wandb.init(
            project="carso-conv-2024",
            config={
                "base_model": f"{BASE_MODEL_NAME} ({MODEL_REFERENCE})",
                "dataset": DATASET_NAME,
                "epochs": args.epochs,
                "batch_size": batchsize * world_size,
                "gpu_count": world_size,
                "n_workers": cpus_per_task,
                "attacks": "FGSM Linf (eps=4/255, 8/255) + PGD Linf (eps=4/255, 8/255; steps=40; alpha=0.01)",
                "adversarial_fraction": args.advfrac,
                "loss_function": "Pixelwise BCE (reduction: sum)",
                "optimizer": f"RAdam + Lookahead ({LAH_STEPS} steps)",
                "lr_scheduler": f"Linear Warmup ({INIT_LR} to {complbr}, in {eup} epochs) + Flat ({complbr}, for {efl} epochs) + Linear Anneal ({complbr} to {compflr}, in {ean} epochs)",  # noqa: E501
            },
        )
    # ──────────────────────────────────────────────────────────────────────────
    carso_machinery.train()
    # ──────────────────────────────────────────────────────────────────────────

    for epoch_idx in trange(args.epochs, desc="Training epoch", disable=(local_rank != 0)):
        if args.dist:
            train_dl.sampler.set_epoch(epoch_idx)  # type: ignore
        # ──────────────────────────────────────────────────────────────────────
        unsc_loss: Tensor = th.zeros((1,), device=device)
        unsc_bce: Tensor = th.zeros((1,), device=device)
        unsc_kld: Tensor = th.zeros((1,), device=device)
        # ──────────────────────────────────────────────────────────────────────
        vae_beta: float = beta_scheduler(epoch_idx, args.epochs)
        # ──────────────────────────────────────────────────────────────────────
        for batch_idx, batched_datapoint in tqdm(
            enumerate(train_dl),
            total=len(train_dl),
            desc="Batch",
            leave=False,
            disable=(local_rank != 0),
        ):
            data, _, old_data = adversarial_apply(
                batched_datapoint,  # type: ignore
                device=device,
                perturbed_fraction=args.advfrac,
                output_also_clean=True,
            )
            # ──────────────────────────────────────────────────────────────────
            optimizer.zero_grad()
            input_reco, cvae_mu, cvae_sigma = carso_machinery(data)
            unsc_loss, unsc_bce, unsc_kld = beta_reco_bce_splitout(
                input_reco, old_data, cvae_mu, cvae_sigma, beta=vae_beta
            )

            loss = unsc_loss * world_size  # DDP averages .grad, compute sum!
            loss.backward()

            optimizer.step()
        # ──────────────────────────────────────────────────────────────────────
        # Log at every epoch
        wandb_loss: Tensor = unsc_loss.detach()
        wandb_bce: Tensor = unsc_bce.detach()
        wandb_kld: Tensor = unsc_kld.detach()
        if args.dist:
            dist.reduce(wandb_loss, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(wandb_bce, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(wandb_kld, dst=0, op=dist.ReduceOp.SUM)

        if args.wandb and local_rank == 0:
            wandb_loss: Tensor = wandb_loss / (batchsize * world_size)
            wandb_bce: Tensor = wandb_bce / (batchsize * world_size)
            wandb_kld: Tensor = wandb_kld / (batchsize * world_size)
            wandb.log(
                {
                    "lr": optimizer.param_groups[0]["lr"],
                    "beta": vae_beta,
                    "loss_tot": wandb_loss.item(),
                    "loss_bce": wandb_bce.item(),
                    "loss_kld": wandb_kld.item(),
                },
                step=epoch_idx,
            )
        # ──────────────────────────────────────────────────────────────────────
        # Step the scheduler
        scheduler.step()
    # ──────────────────────────────────────────────────────────────────────────
    # Model saving / logging to W&B
    if (args.save or args.wandb) and local_rank == 0:
        # Load proper weights if using Lookahead
        if isinstance(optimizer, Lookahead):
            # noinspection PyProtectedMember
            optimizer._backup_and_load_cache()

        save_model(
            (carso_machinery.module.repr_compressors if args.dist else carso_machinery.repr_compressors),
            f"{MODELS_PATH_BASE}{BASE_MODEL_NAME}_{DATASET_NAME}_{MODEL_REFERENCE}_repr_compressors.safetensors",
        )

        save_model(
            (carso_machinery.module.repr_fcn_compressor if args.dist else carso_machinery.repr_fcn_compressor),
            f"{MODELS_PATH_BASE}{BASE_MODEL_NAME}_{DATASET_NAME}_{MODEL_REFERENCE}_repr_fcn_compressor.safetensors",
        )

        save_model(
            (carso_machinery.module.decoder if args.dist else carso_machinery.decoder),
            f"{MODELS_PATH_BASE}{BASE_MODEL_NAME}_{DATASET_NAME}_{MODEL_REFERENCE}_decoder.safetensors",
        )

    if args.wandb and local_rank == 0:
        wandb_repr_compressors = wandb.Artifact(
            f"{BASE_MODEL_NAME}_{DATASET_NAME}_{MODEL_REFERENCE}_repr_compressors",
            type="model",
        )
        wandb_repr_compressors.add_file(
            f"{MODELS_PATH_BASE}{BASE_MODEL_NAME}_{DATASET_NAME}_{MODEL_REFERENCE}_repr_compressors.safetensors"
        )

        wandb_repr_fcn_compressor = wandb.Artifact(
            f"{BASE_MODEL_NAME}_{DATASET_NAME}_{MODEL_REFERENCE}_repr_fcn_compressor",
            type="model",
        )
        wandb_repr_fcn_compressor.add_file(
            f"{MODELS_PATH_BASE}{BASE_MODEL_NAME}_{DATASET_NAME}_{MODEL_REFERENCE}_repr_fcn_compressor.safetensors"
        )

        wandb_decoder = wandb.Artifact(f"{BASE_MODEL_NAME}_{DATASET_NAME}_{MODEL_REFERENCE}_decoder", type="model")
        wandb_decoder.add_file(
            f"{MODELS_PATH_BASE}{BASE_MODEL_NAME}_{DATASET_NAME}_{MODEL_REFERENCE}_decoder.safetensors"
        )

        wandb.log_artifact(wandb_repr_compressors)
        wandb.log_artifact(wandb_repr_fcn_compressor)
        wandb.log_artifact(wandb_decoder)

    # ──────────────────────────────────────────────────────────────────────────
    if args.wandb and local_rank == 0:
        wandb.finish()

    if args.dist:
        dist.destroy_process_group()


# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    main_run(main_parse())


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
# ──────────────────────────────────────────────────────────────────────────────
