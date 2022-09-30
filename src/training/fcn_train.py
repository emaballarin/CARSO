#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---- IMPORTS ----
import argparse

import neptune.new as neptune
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
from torch.optim.lr_scheduler import ReduceLROnPlateau


def main():  # pylint: disable=too-many-locals,too-many-statements # NOSONAR
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
    parser.add_argument(
        "--neptunelog",
        action="store_true",
        default=False,
        help="Log selected metdata to neptune.ai",
    )
    parser.add_argument(
        "--autolr",
        action="store_true",
        default=False,
        help="Automatically schedule learning rate to be recuded on plateau",
    )
    args = parser.parse_args()

    # ---- NEPTUNE ----
    if args.neptunelog:
        run_tags = ["MNIST", "FCN"]
        if args.attack:
            run_tags.append("adversarial")
        else:
            run_tags.append("clean")

        run = neptune.init_run(project="emaballarin/CARSO", tags=run_tags)

    # ---- DEVICE HANDLING ----
    use_cuda = not args.no_cuda and th.cuda.is_available()
    device = th.device("cuda" if use_cuda else "cpu")

    # ---- MODEL DEFINITION / INSTANTIATION ----
    model = mnistfcn_dispatcher(device=device)
    for layer in model.modules():
        mishlayer_init(layer)

    # ---- TRAINING TUNING ----
    TRAIN_BATCHSIZE: int = 256
    TEST_BATCHSIZE: int = 512

    if args.autolr:
        TRAIN_EPOCHS: int = 200
        startlr = 5 * 1e-2
    else:
        TRAIN_EPOCHS: int = 60
        startlr = 1e-2

    LOSSFN = F.nll_loss
    OPTIMIZER = RAdam(model.parameters(), lr=startlr)
    if args.autolr:
        SCHEDULER = ReduceLROnPlateau(
            OPTIMIZER,
            mode="max",
            factor=0.6,
            patience=7,
            cooldown=1,
            verbose=True,
        )
    else:
        SCHEDULER = MultiStepLR(
            OPTIMIZER, milestones=[15, 20, 25, 30, 40, 50], gamma=0.5
        )

    if args.neptunelog:
        run_params = {
            "epoch_nr": TRAIN_EPOCHS,
            "batch_size": TRAIN_BATCHSIZE,
            "optimizer": "RAdam",
            "lr": 1e-2,
            "scheduler": OPTIMIZER.__class__.__name__,
            "scheduler_milestones": [15, 20, 25, 30, 40, 50],
            "scheduler_gamma": 0.5,
            "loss_fn": "nll_loss",
            "architecture": "FCN",
            "architecture_params": {
                "input_size": 784,
                "hidden_sizes": (200, 80),
                "output_size": 10,
                "dropout": (0.15, 0.15, 0.0),
                "activations": "mish",
                "gating": "log_softmax",
                "batchnorm": (True, True, False),
                "bias": True,
                "data_normalization": {"mean": 0.1307, "std": 0.3081},
            },
        }
        run["parameters"] = run_params

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
        train_l, train_a = test(
            model, device, totr_dl, LOSSFN, totr_acc_avgmeter, quiet=False
        )
        run["train/lr"].log(OPTIMIZER.param_groups[0]["lr"])
        run["train/loss"].log(train_l)
        run["train/accuracy"].log(train_a)
        print("\nON TEST SET:")
        test_l, test_a = test(
            model, device, test_dl, LOSSFN, test_acc_avgmeter, quiet=False
        )
        run["test/loss"].log(test_l)
        run["test/accuracy"].log(test_a)
        print("\n\n\n")

        # Tweaks for the Lookahead optimizer (after testing)
        if isinstance(OPTIMIZER, Lookahead):
            OPTIMIZER._clear_and_load_backup()  # pylint: disable=protected-access

        # Scheduling step (outer)
        if args.autolr:
            if args.attack:
                SCHEDULER.step(train_l)
            else:
                SCHEDULER.step(train_a)
        else:
            SCHEDULER.step()  # pylint: disable=no-value-for-parameter

    # ---- SAVE MODEL ----
    if args.save_model or args.neptunelog:
        model_namepath = f"../models/mnist_fcn_{namepiece}.pth"
        th.save(model.state_dict(), model_namepath)

    if args.neptunelog:
        run["model_weights"].upload(model_namepath)
        run.stop()


# Run!
if __name__ == "__main__":
    main()
