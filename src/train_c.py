#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  Copyright (c) 2023 Emanuele Ballarin <emanuele@ballarin.cc>
#  Released under the terms of the MIT License
#  (see: https://url.ballarin.cc/mitlicense)
#
# ------------------------------------------------------------------------------
import argparse
import math

import torch as th
import wandb
from carso import CARSOWrap
from ebtorch.nn import beta_reco_bce
from ebtorch.nn.utils import AdverApply
from ebtorch.optim import ralah_optim
from tooling.architectures import cifar_data_prep_dispatcher
from tooling.architectures_deepmind import PreActResNet
from tooling.attacks import attacks_dispatcher
from tooling.data import cifarten_dataloader_dispatcher
from torch.optim.lr_scheduler import CyclicLR
from torch.optim.swa_utils import AveragedModel as SwaAveragedModel
from torch.optim.swa_utils import update_bn as swa_update_bn
from tqdm.auto import tqdm
from tqdm.auto import trange

# ------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="ResNet18+CARSO on CIFAR10 training")
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
        "--fastswa",
        action="store_true",
        default=False,
        help="Use fast stochastic weight averaging (default: False)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        metavar="<nr_of_epochs>",
        help="Number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=1536,
        metavar="<batch_size>",
        help="Batch size for training (default: 1536)",
    )
    parser.add_argument(
        "--advfrac",
        type=float,
        default=0.4,
        metavar="<adversarial_fraction>",
        help="Fraction of the batch to be adversarially perturbed (default: 0.4)",
    )
    args = parser.parse_args()
    # --------------------------------------------------------------------------

    # Device selection
    use_cuda = th.cuda.is_available()
    device = th.device("cuda" if use_cuda else "cpu")

    # Dataset/DataLoader
    train_dl, _, _ = cifarten_dataloader_dispatcher(
        batch_size_train=args.batchsize,
        batch_size_test=1,
        cuda_accel=device == th.device("cuda"),
    )
    del _

    # ------------------------------------------------------------------------------

    # Models
    vanilla_classifier = PreActResNet()
    vanilla_classifier.load_state_dict(
        th.load("../models/cifar10_linf_resnet18_ddpm.pt")
    )
    vanilla_classifier.to(device).eval()

    carso_machinery = CARSOWrap(
        # Relevant
        wrapped_model=vanilla_classifier,
        input_data_height=32,
        input_data_width=32,
        input_data_channels=3,
        wrapped_repr_size=204810,
        compressed_repr_data_size=512,
        shared_musigma_layer_size=192,
        sampled_code_size=128,
        input_data_no_compress=False,
        input_data_conv_flatten=True,
        repr_data_no_compress=False,
        slim_neck_repr_compressor=True,
        is_deconvolutional_decoder=True,
        is_cifar_decoder=True,
        binarize_repr=False,
        input_preprocessor=cifar_data_prep_dispatcher(device, post_flatten=False),
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
    carso_machinery.to(device).train()

    repr_layers = (
        "layer_0.1.conv_2d_2",
        "layer_1.0.conv_2d_2",
        "layer_1.0.shortcut",
        "layer_2.0.conv_2d_2",
        "layer_2.0.shortcut",
        "layer_2.1.conv_2d_2",
        "layer_3.0.conv_2d_2",
        "layer_3.0.shortcut",
        "layer_3.1.conv_2d_2",
        "logits",
    )

    # Adapt learning rate to batch size (heuristically: linear scaling)
    lr_magic_constant: float = 3.5
    adapted_lr_max: float = lr_magic_constant * 1e-5 * args.batchsize
    adapted_lr_min: float = 0.5e-8

    if args.fastswa:
        swa_ncycles: int = 5
        swa_c: int = max(4, math.ceil(args.epochs / (3 * swa_ncycles)))
        adapted_epochs: int = args.epochs + swa_ncycles * swa_c + 1
    else:
        adapted_epochs: int = args.epochs

    optimizer = ralah_optim(
        carso_machinery.parameters_to_train(), radam_lr=1e-3, la_steps=5
    )

    scheduler = CyclicLR(
        optimizer,
        base_lr=adapted_lr_min,
        max_lr=adapted_lr_max,
        step_size_up=int(1 * args.epochs / 3),
        step_size_down=int(args.epochs - int(1 * args.epochs / 3)),
        cycle_momentum=False,
        mode="triangular",
    )

    if args.fastswa:
        swa_numax: float = adapted_lr_max / 3
        swa_numin: float = swa_numax * 1e-4
        swa_deltanu: float = (swa_numax - swa_numin) / (swa_c - 1)
    else:
        swa_numax: int = 0

    adversaries = attacks_dispatcher(model=vanilla_classifier, dataset="cifarnorm")
    adversarial_apply = AdverApply(adversaries=adversaries)

    # --------------------------------------------------------------------------

    # WandB logging
    if args.wandb:
        wandb.init(
            project="carso-for-neurips-2023",
            config={
                "base_model": "cifar10_linf_resnet18_ddpm (Rebuffi, 2021)",
                "batch_size": args.batchsize,
                "epochs": adapted_epochs,
                "fast-SWA on last 1/4th epochs": args.fastswa,
                "loss_function": "Pixelwise Binary CrossEntropy; Reduction: Sum",
                "optimizer": "RAdam + Lookahead (5 steps)",
                "scheduler": f"CyclicLR (triangular): base_lr={adapted_lr_min}, max_lr={adapted_lr_max}, step_size_up={int(1 * args.epochs / 3)}, step_size_down={int(args.epochs - int(1 * args.epochs / 3))}{' and linear fast-SWA to follow' if args.fastswa else ''}",
                "attacks": "FGSM Linf eps=4/255 eps=8/255 + PGD Linf eps=4/255 eps=8/255 steps=40 alpha=0.01",
                "batchwise_adversarial_fraction": args.advfrac,
            },
        )

    # --------------------------------------------------------------------------
    carso_machinery.train()
    carso_machinery.notify_train_eval_changes(armed=True, hardened=False)
    # --------------------------------------------------------------------------
    if args.fastswa:
        carso_machinery.set_repr_layers_names_lookup(repr_layers)
        swa_carso_machinery = SwaAveragedModel(model=carso_machinery, device=device)
    # --------------------------------------------------------------------------

    print("\nTraining...\n")

    for epoch in trange(adapted_epochs, desc="Training epoch"):
        # ----------------------------------------------------------------------
        for batch_idx, batched_datapoint in tqdm(
            enumerate(train_dl),
            total=len(train_dl),
            desc="Batch within epoch",
            leave=False,
        ):
            batched_datapoint = adversarial_apply(
                batched_datapoint,
                device=device,
                perturbed_fraction=args.advfrac,
                output_also_clean=True,
            )

            data, target, old_data = batched_datapoint

            optimizer.zero_grad()

            input_reco, (cvae_mu, cvae_sigma) = carso_machinery(data, repr_layers)
            loss = beta_reco_bce(input_reco, old_data, cvae_mu, cvae_sigma)
            loss.backward()

            # Optimize
            optimizer.step()

        # ------------------------------------------------------------------------------
        # Every epoch
        if args.wandb:
            wandb.log({"lr": optimizer.param_groups[0]["lr"], "loss": loss.item()})
        # ----------------------------------------------------------------------

        if (not args.fastswa) or (epoch < args.epochs):
            scheduler.step()
        else:
            # Include weights in average
            optimizer._backup_and_load_cache()
            swa_carso_machinery.update_parameters(carso_machinery)
            optimizer._clear_and_load_backup()
            # Update LR manually
            if (epoch - args.epochs) % swa_c == 0:
                optimizer.param_groups[0]["lr"] = swa_numax
            else:
                optimizer.param_groups[0]["lr"] = (
                    optimizer.param_groups[0]["lr"] - swa_deltanu
                )

    # --------------------------------------------------------------------------
    if args.save_model or args.wandb:
        model_namepath_compressor = (
            "../models/carso_reprcompressor_rebuffirn18_cifar10_adv.pth"
        )
        model_namepath_dec = "../models/carso_dec_rebuffirn18_cifar10_adv.pth"

        if args.fastswa:
            swa_update_bn(loader=train_dl, model=swa_carso_machinery, device=device)
            th.save(
                swa_carso_machinery.module.repr_compressor.state_dict(),
                model_namepath_compressor,
            )
            th.save(swa_carso_machinery.module.dec.state_dict(), model_namepath_dec)
        else:
            optimizer._backup_and_load_cache()
            th.save(
                carso_machinery.repr_compressor.state_dict(),
                model_namepath_compressor,
            )
            th.save(carso_machinery.dec.state_dict(), model_namepath_dec)

    if args.wandb:
        repr_compressor = wandb.Artifact(
            "carso_reprcompressor_rebuffirn18_cifar10_adv", type="model"
        )
        carso_dec = wandb.Artifact("carso_dec_rebuffirn18_cifar10_adv", type="model")
        repr_compressor.add_file(model_namepath_compressor)
        carso_dec.add_file(model_namepath_dec)
        wandb.log_artifact(repr_compressor)
        wandb.log_artifact(carso_dec)

    # --------------------------------------------------------------------------
    if args.wandb:
        wandb.finish()


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
