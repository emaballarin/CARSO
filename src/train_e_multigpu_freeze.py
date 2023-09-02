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
import torch.distributed as dist
import wandb
from carso import CARSOWrap
from ebtorch.data import cifarhundred_dataloader_dispatcher
from ebtorch.data import data_prep_dispatcher_3ch
from ebtorch.distributed import slurm_nccl_env
from ebtorch.nn import beta_reco_bce
from ebtorch.nn import WideResNet
from ebtorch.nn.utils import AdverApply
from ebtorch.optim import Lookahead
from ebtorch.optim import onecycle_linlin
from ebtorch.optim import ralah_optim
from tooling.attacks import attacks_dispatcher
from torch.nn.parallel import DistributedDataParallel as DiDaPar
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from tqdm.auto import trange


# ------------------------------------------------------------------------------
def main_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WideResNet-28-10+CARSO on CIFAR100 Multi-GPU Training"
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        default=False,
        help="Save model after training (default: False)",
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
        default=150,
        metavar="<nr_of_epochs>",
        help="Number of epochs to train (default: 150)",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=1280,
        metavar="<batch_size>",
        help="Per-GPU batch size for training (default: 1280)",
    )
    parser.add_argument(
        "--advfrac",
        type=float,
        default=0.4,
        metavar="<adversarial_fraction>",
        help="Fraction of the batch to be adversarially perturbed (default: 0.4)",
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
    # Repeated twice just to allow gathering of dataset for DistributedSampler
    train_dl, _, _ = cifarhundred_dataloader_dispatcher(
        batch_size_train=args.batchsize,
        batch_size_test=1,
        cuda_accel=True,
        shuffle_test=False,
        unshuffle_train=True,
    )
    train_dl, _, _ = cifarhundred_dataloader_dispatcher(
        batch_size_train=args.batchsize,
        batch_size_test=1,
        cuda_accel=True,
        shuffle_test=False,
        unshuffle_train=True,
        dataloader_kwargs={
            "sampler": DistributedSampler(train_dl.dataset),
            "num_workers": cpus_per_task,
        },
    )
    del _

    # --------------------------------------------------------------------------

    # Models
    vanilla_classifier = WideResNet(num_classes=100, bn_momentum=0.01)
    vanilla_classifier.load_state_dict(th.load("../models/cifar100_a5_b12_t4_50m_w.pt"))
    vanilla_classifier.to(device).eval()

    carso_machinery = CARSOWrap(
        # Relevant
        wrapped_model=vanilla_classifier,
        input_data_height=32,
        input_data_width=32,
        input_data_channels=3,
        wrapped_repr_size=532580,
        compressed_repr_data_size=2048,
        shared_musigma_layer_size=128,
        sampled_code_size=96,
        input_data_no_compress=False,
        input_data_conv_flatten=True,
        repr_data_no_compress=False,
        slim_neck_repr_compressor=True,
        is_deconvolutional_decoder=True,
        is_cifar_decoder=10,
        binarize_repr=False,
        input_preprocessor=data_prep_dispatcher_3ch(device, post_flatten=False),
        # Forced/Dummy
        compressed_input_data_size=0,
        ensemble_numerosity=0,
        convolutional_input_compressor=False,
        differentiable_inference=False,
        sum_of_softmaxes_inference=True,
        suppress_stochastic_inference=True,
        output_logits=False,
        headless_mode=False,
    )
    carso_machinery = th.nn.SyncBatchNorm.convert_sync_batchnorm(carso_machinery)
    carso_machinery.to(device)
    carso_machinery = DiDaPar(
        carso_machinery,
        device_ids=[local_rank],
        find_unused_parameters=True,
        gradient_as_bucket_view=True,
    )
    carso_machinery.train()

    repr_layers = (
        "layer.0.block.1.conv_1",
        "layer.1.block.0.shortcut",
        "layer.1.block.1.conv_1",
        "layer.1.block.2.conv_1",
        "layer.2.block.0.shortcut",
        "layer.2.block.1.conv_1",
        "layer.2.block.3.conv_1",
        "logits",
    )

    optimizer = ralah_optim(carso_machinery.parameters(), radam_lr=0.0, la_steps=6)

    optimizer, scheduler = onecycle_linlin(
        optim=optimizer,
        init_lr=5e-9,
        max_lr=0.04,  # (4.0 / 8.0) * 1.0e-4 * args.batchsize * world_size,
        final_lr=1.25e-8 * args.batchsize * world_size,
        up_frac=0.25,
        total_steps=args.epochs,
    )

    adversaries = attacks_dispatcher(model=vanilla_classifier, dataset="cifarnorm")
    adversarial_apply = AdverApply(adversaries=adversaries)

    # --------------------------------------------------------------------------

    # WandB logging
    if args.wandb and local_rank == 0:
        wandb.init(
            project="carso-for-neurips-2023",
            config={
                "base_model": "cifar100_linf_wrn2810_a5_b12_t4_50m (Cui, 2023)",
                "batch_size": args.batchsize * world_size,
                "per_gpu_batch_size": args.batchsize,
                "gpu_count": world_size,
                "epochs": args.epochs,
                "loss_function": "Pixelwise Binary CrossEntropy; Reduction: Sum",
                "optimizer": "RAdam + Lookahead (5 steps)",
                "scheduler": "See the code!",
                "attacks": "FGSM Linf eps=4/255 eps=8/255 + PGD Linf eps=4/255 eps=8/255 steps=40 alpha=0.01",
                "batchwise_adversarial_fraction": args.advfrac,
            },
        )

    # --------------------------------------------------------------------------
    carso_machinery.train()

    for epoch_idx in trange(args.epochs, desc="Training epoch", disable=(local_rank != 0)):  # type: ignore
        # ----------------------------------------------------------------------
        # Every epoch
        train_dl.sampler.set_epoch(epoch_idx)  # type: ignore
        for batch_idx, batched_datapoint in tqdm(  # type: ignore
            enumerate(train_dl),
            total=len(train_dl),
            desc="Batch within epoch",
            leave=False,
            disable=(local_rank != 0),
        ):
            batched_datapoint = adversarial_apply(
                batched_datapoint,
                device=device,
                perturbed_fraction=args.advfrac,
                output_also_clean=True,
            )

            data, _, old_data = batched_datapoint

            optimizer.zero_grad()

            input_reco, (cvae_mu, cvae_sigma) = carso_machinery(data, repr_layers)
            loss = beta_reco_bce(input_reco, old_data, cvae_mu, cvae_sigma)
            loss = loss * world_size  # DDP averages grad, we want sum!
            loss.backward()

            # Optimize
            optimizer.step()

        # Every epoch
        # ----------------------------------------------------------------------
        loss_to_log = loss.detach().clone()
        dist.reduce(loss_to_log, dst=0, op=dist.ReduceOp.SUM)
        if args.wandb and local_rank == 0:
            loss_to_log = loss_to_log / (world_size**2 * args.batchsize)
            wandb.log(
                {
                    "lr": optimizer.param_groups[0]["lr"],
                    "avg. loss": loss_to_log.item(),
                }
            )
        # ----------------------------------------------------------------------
        scheduler.step()
        # ----------------------------------------------------------------------

    # --------------------------------------------------------------------------
    if (args.save_model or args.wandb) and local_rank == 0:
        model_namepath_compressor = (
            "../models/carso_reprcompressor_cuiwrn2810_cifar100_adv.pth"
        )
        model_namepath_dec = "../models/carso_dec_cuiwrn2810_cifar100_adv.pth"
        if isinstance(optimizer, Lookahead):
            optimizer._backup_and_load_cache()
        th.save(
            carso_machinery.module.repr_compressor.state_dict(),
            model_namepath_compressor,
        )
        th.save(carso_machinery.module.dec.state_dict(), model_namepath_dec)

    if args.wandb and local_rank == 0:
        repr_compressor = wandb.Artifact(
            "carso_reprcompressor_cuiwrn2810_cifar100_adv", type="model"
        )
        carso_dec = wandb.Artifact("carso_dec_cuiwrn2810_cifar100_adv", type="model")
        repr_compressor.add_file(model_namepath_compressor)
        carso_dec.add_file(model_namepath_dec)
        wandb.log_artifact(repr_compressor)
        wandb.log_artifact(carso_dec)

    # --------------------------------------------------------------------------
    if args.wandb and local_rank == 0:
        wandb.finish()

    # --------------------------------------------------------------------------

    dist.destroy_process_group()


# ------------------------------------------------------------------------------


def main() -> None:
    parser_output = main_parse()
    main_run(parser_output)


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
