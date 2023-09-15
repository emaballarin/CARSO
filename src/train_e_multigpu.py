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
from ebtorch.nn.utils import subset_state_dict
from ebtorch.optim import Lookahead
from ebtorch.optim import onecycle_linlin
from ebtorch.optim import ralah_optim
from tooling.attacks import attacks_dispatcher
from torch.distributed.fsdp import CPUOffload
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDParallel
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy as auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDParallel
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
        "--fsdp",
        action="store_true",
        default=False,
        help="Train with Fully-Sharded Data Parallelism (default: False)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        metavar="<nr_of_epochs>",
        help="Number of epochs to train (default: 200)",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=800,
        metavar="<batch_size>",
        help="Per-GPU batch size for training (default: 800)",
    )
    parser.add_argument(
        "--advfrac",
        type=float,
        default=0.12,
        metavar="<adversarial_fraction>",
        help="Fraction of the batch to be adversarially perturbed (default: 0.12)",
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
        wrapped_repr_size=286820,
        compressed_repr_data_size=2816,
        shared_musigma_layer_size=192,
        sampled_code_size=128,
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

    if not args.fsdp:
        carso_machinery = th.nn.SyncBatchNorm.convert_sync_batchnorm(carso_machinery)
        carso_machinery.to(device)
        carso_machinery = DDParallel(
            carso_machinery,
            device_ids=[local_rank],
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )
    else:
        carso_machinery = FSDParallel(
            module=carso_machinery,
            auto_wrap_policy=auto_wrap_policy,
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
            cpu_offload=CPUOffload(offload_params=False),
            limit_all_gathers=False,
            sync_module_states=True,
            use_orig_params=True,
            device_id=local_rank,
        )

    carso_machinery.train()

    repr_layers = (
        "layer.1.block.0.conv_0",  # From: 04/09
        "layer.1.block.1.conv_1",  # From: 04/09
        "layer.2.block.0.conv_1",  # From: 04/09
        "layer.2.block.1.conv_1",  # From: ADD
        "layer.2.block.2.conv_1",  # From: 04/09
        "logits",  # From: 04/09
    )

    optimizer = ralah_optim(
        carso_machinery.parameters(),
        radam_lr=0.0,
        la_steps=6,
        radam_betas=(0.9, 0.99),
    )

    optimizer, scheduler = onecycle_linlin(
        optim=optimizer,
        init_lr=5e-9,
        max_lr=0.065,
        final_lr=1.25e-8 * args.batchsize * world_size,
        up_frac=0.25,
        total_steps=args.epochs,
    )

    adversaries = attacks_dispatcher(model=vanilla_classifier, dataset="cifarnorm")
    adversarial_apply = AdverApply(adversaries=adversaries)

    # --------------------------------------------------------------------------

    # WandB logging
    if args.wandb and rank == 0:
        wandb.init(
            project="carso-for-neurips-2023",
            config={
                "base_model": "cifar100_linf_wrn2810_a5_b12_t4_50m (Cui, 2023)",
                "batch_size": args.batchsize * world_size,
                "per_gpu_batch_size": args.batchsize,
                "gpu_count": world_size,
                "epochs": args.epochs,
                "loss_function": "Pixelwise Binary CrossEntropy; Reduction: Sum",
                "optimizer": "RAdam + Lookahead (6 steps)",
                "scheduler": "See the code!",
                "attacks": "FGSM Linf eps=4/255 eps=8/255 + PGD Linf eps=4/255 eps=8/255 steps=40 alpha=0.01",
                "batchwise_adversarial_fraction": args.advfrac,
            },
        )

    # --------------------------------------------------------------------------
    carso_machinery.train()

    for epoch_idx in trange(args.epochs, desc="Training epoch", disable=(rank != 0)):  # type: ignore
        # ----------------------------------------------------------------------
        # Every epoch
        train_dl.sampler.set_epoch(epoch_idx)  # type: ignore
        for batch_idx, batched_datapoint in tqdm(  # type: ignore
            enumerate(train_dl),
            total=len(train_dl),
            desc="Batch within epoch",
            leave=False,
            disable=(rank != 0),
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
        if args.wandb and rank == 0:
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
    if isinstance(optimizer, Lookahead):
        optimizer._backup_and_load_cache()
    # Handle FSDP case (1)
    if args.fsdp:
        dist.barrier()
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDParallel.state_dict_type(
            carso_machinery, StateDictType.FULL_STATE_DICT, save_policy
        ):
            state_to_save_whole = carso_machinery.state_dict()
    if (args.save_model or args.wandb) and rank == 0:
        model_namepath_compressor = (
            "../models/carso_reprcompressor_cuiwrn2810_cifar100_adv.pth"
        )
        model_namepath_dec = "../models/carso_dec_cuiwrn2810_cifar100_adv.pth"
        # Handle FSDP case (2)
        if args.fsdp:
            th.save(
                subset_state_dict(state_to_save_whole, "repr_compressor"),
                model_namepath_compressor,
            )
            th.save(subset_state_dict(state_to_save_whole, "dec"), model_namepath_dec)
            del state_to_save_whole
        else:
            th.save(
                carso_machinery.module.repr_compressor.state_dict(),
                model_namepath_compressor,
            )
            th.save(carso_machinery.module.dec.state_dict(), model_namepath_dec)

    if args.wandb and rank == 0:
        repr_compressor = wandb.Artifact(
            "carso_reprcompressor_cuiwrn2810_cifar100_adv", type="model"
        )
        carso_dec = wandb.Artifact("carso_dec_cuiwrn2810_cifar100_adv", type="model")
        repr_compressor.add_file(model_namepath_compressor)
        carso_dec.add_file(model_namepath_dec)
        wandb.log_artifact(repr_compressor)
        wandb.log_artifact(carso_dec)

    # --------------------------------------------------------------------------
    if args.wandb and rank == 0:
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
