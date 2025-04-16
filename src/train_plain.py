#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
import argparse
from math import pow as mpow
from typing import Tuple

import torch as th
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import wandb
from ebtorch.data import cifarhundred_dataloader_dispatcher
from ebtorch.data import cifarten_dataloader_dispatcher
from ebtorch.distributed import slurm_nccl_env
from ebtorch.nn import WideResNet
from ebtorch.nn.architectures_resnets_dm import CIFAR100_MEAN
from ebtorch.nn.architectures_resnets_dm import CIFAR100_STD
from ebtorch.nn.architectures_resnets_dm import CIFAR10_MEAN
from ebtorch.nn.architectures_resnets_dm import CIFAR10_STD
from ebtorch.nn.utils import eval_model_on_test
from ebtorch.nn.utils import TelegramBotEcho as TBE
from safetensors.torch import save_model
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from tqdm.auto import trange


# ──────────────────────────────────────────────────────────────────────────────
MODEL_NAME: str = "WRN_28_10"
DATASET_NAME: str = "cifar-"
REF_LR: float = 2e-3
SINGLE_GPU_WORKERS: int = 16


# ──────────────────────────────────────────────────────────────────────────────
def main_parse() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=f"Training {MODEL_NAME} on {DATASET_NAME}*"
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
        "--tgnotif",
        action="store_true",
        default=False,
        help="Notify via Telegram when the training starts and ends (default: False)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        metavar="<epochs>",
        help="Number of epochs to train for (default: 200)",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=128,
        metavar="<batch_size>",
        help="Batch size for training (default: 128)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifarten",
        metavar="<dataset>",
        help="Dataset to train on (default: cifarten)",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
def main_run(args: argparse.Namespace) -> None:

    # Distributed setup / Device selection
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

        # ──────────────────────────────────────────────────────────────────────

    # Telegram notification machinery
    if args.tgnotif and local_rank == 0:
        ebdltgb = TBE("EBDL_TGB_TOKEN", "EBDL_TGB_CHTID")
        ebdltgb.send(f"Training started ({args.dataset})!")

    # Data loading
    batchsize: int = max(args.batchsize // world_size, 2)

    if args.dataset == "cifarten":
        dataset_dispatcher = cifarten_dataloader_dispatcher
        datamean: Tuple[float, float, float] = CIFAR10_MEAN
        datastd: Tuple[float, float, float] = CIFAR10_STD
        nclasses: int = 10
    elif args.dataset == "cifarhundred":
        datamean: Tuple[float, float, float] = CIFAR100_MEAN
        datastd: Tuple[float, float, float] = CIFAR100_STD
        nclasses: int = 100
        dataset_dispatcher = cifarhundred_dataloader_dispatcher
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    train_dl, test_dl, _ = dataset_dispatcher(
        batch_size_train=batchsize,
        batch_size_test=4 * batchsize,
        cuda_accel=(device == th.device("cuda") or args.dist),
        unshuffle_train=args.dist,
        dataloader_kwargs=(
            {"num_workers": cpus_per_task, "persistent_workers": True}
            if not args.dist
            else {}
        ),
    )

    if args.dist:
        train_dl, _, _ = dataset_dispatcher(
            batch_size_train=batchsize,
            batch_size_test=1,
            cuda_accel=True,
            unshuffle_train=True,
            dataloader_kwargs={
                "sampler": DistributedSampler(train_dl.dataset),
                "num_workers": cpus_per_task,
                "persistent_workers": True,
            },
        )
        _, test_dl, _ = dataset_dispatcher(
            batch_size_train=1,
            batch_size_test=4 * batchsize,
            cuda_accel=True,
            unshuffle_train=True,
            dataloader_kwargs={
                "sampler": DistributedSampler(test_dl.dataset),
                "num_workers": cpus_per_task,
                "persistent_workers": True,
            },
        )
        del _

    # Model instantiation
    model: nn.Module = (
        WideResNet(num_classes=nclasses, mean=datamean, std=datastd).to(device).train()
    )
    if args.dist:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.to(device)
        model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
        model.train()

    # Criterion definition
    criterion = lambda x, y: F.cross_entropy(x, y, reduction="mean")  # noqa: E731

    # Optimizer instantiation
    optimizer: Optimizer = SGD(
        params=model.parameters(),
        lr=REF_LR,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True,
        fused=(device == th.device("cuda") or args.dist),
    )

    # LR scheduler instantiation
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=(
            lambda epoch: mpow(
                0.275,
                3 if epoch > 160 else 2 if epoch > 120 else 1 if epoch > 60 else 0,
            )
        ),
    )  # NOSONAR

    # Wandb initialization
    if args.wandb and local_rank == 0:
        wandb.init(
            project="semigood-baselines",
            config={
                "model": MODEL_NAME,
                "dataset": args.dataset,
                "batch_size": batchsize * world_size,
                "epochs": args.epochs,
                "ref_lr": REF_LR,
            },
        )

    # ──────────────────────────────────────────────────────────────────────────
    # TRAINING LOOP
    # ──────────────────────────────────────────────────────────────────────────
    unsc_loss: Tensor = th.tensor(0.0, device=device)
    for eidx in trange(args.epochs, desc="Training epoch", disable=(local_rank != 0)):

        if args.dist:
            train_dl.sampler.set_epoch(eidx)  # type: ignore

        # Training
        for _, (batched_x, batched_y) in tqdm(
            iterable=enumerate(train_dl),
            total=len(train_dl),
            leave=False,
            desc="Training batch",
            disable=(local_rank != 0),
        ):
            batched_x: Tensor = batched_x.to(device)
            batched_y: Tensor = batched_y.to(device)
            optimizer.zero_grad()
            batched_yhat: Tensor = model(batched_x)
            unsc_loss: Tensor = criterion(batched_yhat, batched_y)
            loss = unsc_loss * world_size  # DDP averages .grad, compute sum!
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Evaluation
        testacc: float = eval_model_on_test(
            model=model,
            test_data_loader=test_dl,
            device=th.device(device),
            verbose=True,
        )
        # ──────────────────────────────────────────────────────────────────────

        # Wandb logging
        if args.wandb:
            wandb_loss: Tensor = unsc_loss.detach()
            wandb_acc: Tensor = th.tensor(testacc, device=device).detach()

            if args.dist:
                dist.reduce(wandb_loss, dst=0, op=dist.ReduceOp.SUM)
                dist.reduce(wandb_acc, dst=0, op=dist.ReduceOp.SUM)

            if local_rank == 0:
                wandb_loss /= world_size
                wandb_acc /= world_size
                wandb.log(
                    {
                        "lr": optimizer.param_groups[0]["lr"],
                        "train_loss": wandb_loss.item(),
                        "test_acc": wandb_acc.item(),
                    },
                    step=eidx,
                )
    # ──────────────────────────────────────────────────────────────────────────
    # Model saving
    if args.save and local_rank == 0:
        save_model(
            model.module if args.dist else model,
            f"{MODEL_NAME}_{args.dataset}.safetensors",
        )

    # ──────────────────────────────────────────────────────────────────────────
    if args.wandb and local_rank == 0:
        wandb.finish()
    # ──────────────────────────────────────────────────────────────────────────
    if args.dist:
        dist.destroy_process_group()
    # ──────────────────────────────────────────────────────────────────────────
    if args.tgnotif and local_rank == 0:
        # noinspection PyUnboundLocalVariable
        facc = wandb_acc.item() if args.wandb else "N.A."
        # noinspection PyUnboundLocalVariable
        ebdltgb.send(f"Training ended ({args.dataset})!\nFinal accuracy:{facc}")


# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    main_run(main_parse())


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
# ──────────────────────────────────────────────────────────────────────────────
