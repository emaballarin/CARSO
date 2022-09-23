#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---- IMPORTS ----
import argparse

import torch as th
import torch.nn.functional as F
from ebtorch.logging import AverageMeter
from ebtorch.nn import mishlayer_init
from ebtorch.optim import Lookahead
from ebtorch.optim import RAdam
from tooling.architectures import mnistfcn_dispatcher
from tooling.attacks import attacks_dispatcher
from tooling.data import mnist_dataloader_dispatcher
from tooling.loops import test
from tooling.loops import train_epoch
from torch.optim.lr_scheduler import MultiStepLR


def main():  # pylint: disable=too-many-locals
    # Argument parsing...
    parser = argparse.ArgumentParser(description="FCN on MNIST training")
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress printing during training",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=False,
        help="Disables CUDA acceleration during training",
    )
    parser.add_argument(
        "--attack",
        action="store_true",
        default=False,
        help="Perform iterative adversarial training",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="Save model after training",
    )
    args = parser.parse_args()

    # ---- DEVICE HANDLING ----
    use_cuda = not args.no_cuda and th.cuda.is_available()
    device = th.device("cuda" if use_cuda else "cpu")

    # ---- MODEL DEFINITION / INSTANTIATION ----
    model = mnistfcn_dispatcher(device=device)
    for layer in model.modules():
        mishlayer_init(layer)

    # ---- TRAINING TUNING ----
    TRAIN_BATCHSIZE: int = 128
    TEST_BATCHSIZE: int = 512
    TRAIN_EPOCHS: int = 50
    LOSSFN = F.nll_loss
    OPTIMIZER = RAdam(model.parameters(), lr=1e-2)
    SCHEDULER = MultiStepLR(OPTIMIZER, milestones=[15, 20, 25, 30, 40, 45], gamma=0.5)

    # ---- DATASETS ----
    train_dl, test_dl, totr_dl = mnist_dataloader_dispatcher(
        batch_size_train=TRAIN_BATCHSIZE,
        batch_size_test=TEST_BATCHSIZE,
        cuda_accel=bool(device == th.device("cuda")),
    )

    # ---- TRAINING STATISTICS ----
    train_acc_avgmeter = AverageMeter("batchwise training loss")
    test_acc_avgmeter = AverageMeter("epochwise testing loss")
    totr_acc_avgmeter = AverageMeter("epochwise training loss")

    if args.attack:
        adversaries = attacks_dispatcher(model=model)
        namepiece: str = "adv"
    else:
        adversaries = []
        namepiece: str = "clean"

    # ---- TRAINING LOOP ----
    for epoch in range(1, TRAIN_EPOCHS + 1):

        # Training
        print("TRAINING...")

        train_epoch(
            model=model,
            device=device,
            train_loader=train_dl,
            loss_fn=LOSSFN,
            optimizer=OPTIMIZER,
            epoch=epoch,
            print_every_nep=15,
            train_acc_avgmeter=train_acc_avgmeter,
            inner_scheduler=None,
            quiet=args.quiet,
            adversaries=adversaries,
        )

        # Tweaks for the Lookahead optimizer (before testing)
        if isinstance(OPTIMIZER, Lookahead):
            OPTIMIZER._backup_and_load_cache()  # pylint: disable=protected-access

        # Testing: on training and testing set
        print("\n")
        print("TESTING...")
        print("\nON TRAINING SET:")
        _ = test(model, device, totr_dl, LOSSFN, totr_acc_avgmeter, quiet=False)
        del _
        print("\nON TEST SET:")
        _ = test(model, device, test_dl, LOSSFN, test_acc_avgmeter, quiet=False)
        del _
        print("\n\n\n")

        # Tweaks for the Lookahead optimizer (after testing)
        if isinstance(OPTIMIZER, Lookahead):
            OPTIMIZER._clear_and_load_backup()  # pylint: disable=protected-access

        # Scheduling step (outer)
        SCHEDULER.step()

    # ---- SAVE MODEL ----
    if args.save_model:
        th.save(model.state_dict(), "../../models/mnistfcn_" + namepiece + ".pth")


# Run!
if __name__ == "__main__":
    main()
