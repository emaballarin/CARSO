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
import torch.nn.functional as F
import wandb
from ebtorch.data import fashionmnist_dataloader_dispatcher
from ebtorch.data import mnist_dataloader_dispatcher
from ebtorch.logging import AverageMeter
from ebtorch.nn import mishlayer_init
from ebtorch.nn.utils import AdverApply
from ebtorch.optim import Lookahead
from ebtorch.optim import ralah_optim
from tooling.architectures import fashionmnist_cnn_classifier_dispatcher
from tooling.architectures import mnist_cnn_classifier_dispatcher
from tooling.architectures import mnist_fcn_classifier_dispatcher
from tooling.attacks import attacks_dispatcher
from torch.optim.lr_scheduler import CyclicLR
from tqdm.auto import tqdm
from tqdm.auto import trange


def main():  # NOSONAR
    parser = argparse.ArgumentParser(
        description="FCN/CNN on MNIST/FashionMNIST clean/adversarial training"
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
        default=50,
        metavar="<nr_of_epochs>",
        help="Number of epochs to train (default: 25)",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=256,
        metavar="<batch_size>",
        help="Batch size for training (default: 256)",
    )
    parser.add_argument(
        "--do_attack",
        action="store_true",
        default=False,
        help="Train the model adversarially (default: False)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="fcn",
        metavar="<model_type>",
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

    # Device selection
    use_cuda = th.cuda.is_available()
    device = th.device("cuda" if use_cuda else "cpu")
    # --------------------------------------------------------------------------

    # Dataset selection
    if args.dataset == "mnist":
        data_dispatcher = mnist_dataloader_dispatcher
    elif args.dataset == "fashionmnist":
        data_dispatcher = fashionmnist_dataloader_dispatcher
    else:
        raise ValueError("Invalid dataset selected! Valid options: mnist, fashionmnist")

    train_dl, test_dl, _ = data_dispatcher(
        batch_size_train=args.batchsize,
        batch_size_test=2 * args.batchsize,
        cuda_accel=device == th.device("cuda"),
    )
    del _
    # --------------------------------------------------------------------------

    # Model selection
    if args.model_type == "fcn":
        classifier_model = mnist_fcn_classifier_dispatcher(kwta_filter=False)
    elif args.model_type == "cnn":
        if args.dataset == "mnist":
            classifier_model = mnist_cnn_classifier_dispatcher()
        elif args.dataset == "fashionmnist":
            classifier_model = fashionmnist_cnn_classifier_dispatcher()
        else:
            raise ValueError(
                "Invalid dataset selected! Valid options: mnist, fashionmnist"
            )
    else:
        raise ValueError("Invalid model selected! Valid options: fcn, cnn")

    # Proper initialization of weights
    classifier_model.to(device)
    for layer in classifier_model.modules():
        mishlayer_init(layer)

    # --------------------------------------------------------------------------
    # Adversaries selection
    adversaries = attacks_dispatcher(model=classifier_model, fgsm=False)
    adversarial_apply = AdverApply(adversaries=adversaries)
    # --------------------------------------------------------------------------

    # Training setup
    optimizer = ralah_optim(classifier_model.parameters(), radam_lr=5e-2, la_steps=5)
    epochs_up = 2 * args.epochs // 5
    scheduler = CyclicLR(
        optimizer,
        base_lr=1e-8,
        max_lr=5e-3,
        step_size_up=epochs_up,
        step_size_down=args.epochs - epochs_up,
        cycle_momentum=False,
        mode="triangular",
    )
    loss_fn = F.nll_loss
    # --------------------------------------------------------------------------
    # Logging
    train_loss_meter = AverageMeter(name="training loss")
    test_acc_meter = AverageMeter(name="test accuracy")

    # WandB logging
    advpiece = "adversarial" if args.do_attack else "clean"
    if args.wandb:
        wandb.init(
            project="carso-for-neurips-2023",
            config={
                "base_model": f"The {advpiece} {args.model_type} training right now",
                "batch_size": args.batchsize,
                "epochs": args.epochs,
                "loss_function": "Negative Log Likelihood",
                "optimizer": "RAdam + Lookahead (3 steps)",
                "scheduler": f"CyclicLR (triangular): base_lr=1e-8, max_lr=5e-3, epochs_up={epochs_up}, epochs_down={args.epochs - epochs_up}",
                "attacks": "PGD Linf eps=4/255 eps=8/255 steps=40 alpha=0.01",
            },
        )
    # --------------------------------------------------------------------------
    for _ in trange(args.epochs, desc="Training epoch"):  # type: ignore
        # USUAL TRAINING LOOP (cfr. tooling/loops.py)
        classifier_model.train()
        train_loss_meter.reset()

        for batch_idx, batched_datapoint in tqdm(  # type: ignore
            enumerate(train_dl), total=len(train_dl), leave=False, desc="Training batch"
        ):
            batched_datapoint = adversarial_apply(
                batched_datapoint,
                device=device,
                perturbed_fraction=1.0 if args.do_attack else 0.0,
                output_also_clean=False,
            )

            data, target = batched_datapoint

            # BackProp + Optim
            optimizer.zero_grad()
            output = classifier_model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            train_loss_meter.update(loss.item())

        # USUAL TEST LOOP (cfr. tooling/loops.py)
        if isinstance(optimizer, Lookahead):
            optimizer._backup_and_load_cache()

        classifier_model.eval()
        test_acc_meter.reset()
        correct: int = 0
        with th.no_grad():
            for data, target in tqdm(  # type: ignore
                iterable=test_dl, desc="Testing batch", leave=False
            ):
                data, target = data.to(device), target.to(device)
                output = classifier_model(data)
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max (log-)probability class
                correct += pred.eq(target.view_as(pred)).sum().item()

        ltlds: int = len(test_dl.dataset)  # type: ignore
        test_acc_meter.update(correct / ltlds)

        if isinstance(optimizer, Lookahead):
            optimizer._clear_and_load_backup()

        # Every epoch
        if args.wandb:
            wandb.log(
                {
                    "lr": optimizer.param_groups[0]["lr"],
                    "trainloss": train_loss_meter.avg,
                    "testacc": test_acc_meter.avg,
                }
            )
        # ----------------------------------------------------------------------

        scheduler.step()

    # Model saving
    if args.save_model:
        advname = "adv" if args.do_attack else "clean"
        namepiece = f"{args.model_type}_{args.dataset}_{advname}"
        savepath = f"../models/{namepiece}.pth"

        if isinstance(optimizer, Lookahead):
            optimizer._backup_and_load_cache()

        th.save(classifier_model.state_dict(), savepath)

        if args.wandb:
            wandb_classif_model_object = wandb.Artifact(namepiece, type="model")
            wandb_classif_model_object.add_file(savepath)
            wandb.log_artifact(wandb_classif_model_object)

    if args.wandb:
        wandb.finish()


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
