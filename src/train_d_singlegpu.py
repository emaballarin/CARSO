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
import wandb
from carso import CARSOWrap
from ebtorch.data import cifarten_dataloader_dispatcher
from ebtorch.data import data_prep_dispatcher_3ch
from ebtorch.nn import beta_reco_bce
from ebtorch.nn import WideResNet
from ebtorch.nn.utils import AdverApply
from ebtorch.optim import Lookahead
from ebtorch.optim import ralah_optim
from ebtorch.optim import tricyc1c
from tooling.attacks import attacks_dispatcher
from tqdm.auto import tqdm
from tqdm.auto import trange

# ------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WideResNet-28-10+CARSO on CIFAR10 Training"
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
        help="Number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=800,
        metavar="<batch_size>",
        help="Batch size for training (default: 800)",
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
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    # Dataset/DataLoader
    train_dl, _, _ = cifarten_dataloader_dispatcher(
        batch_size_train=args.batchsize,
        batch_size_test=1,
        cuda_accel=device == th.device("cuda"),
    )
    del _

    # ------------------------------------------------------------------------------

    # Models
    vanilla_classifier = WideResNet(bn_momentum=0.01)
    vanilla_classifier.load_state_dict(th.load("../models/cifar10_a3_b10_t4_20m_w.pt"))
    vanilla_classifier.to(device).eval()

    carso_machinery = CARSOWrap(
        # Relevant
        wrapped_model=vanilla_classifier,
        input_data_height=32,
        input_data_width=32,
        input_data_channels=3,
        wrapped_repr_size=573450,
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
    carso_machinery.to(device).train()

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

    # Adapt learning rate to batch size (heuristically: linear scaling)
    lr_magic_constant: float = 3.5
    adapted_lr_max: float = lr_magic_constant * 1e-5 * args.batchsize
    adapted_lr_min: float = 0.5e-8

    optimizer = ralah_optim(
        carso_machinery.parameters_to_train(), radam_lr=1e-3, la_steps=5
    )

    optimizer, scheduler = tricyc1c(
        optimizer, adapted_lr_min, adapted_lr_max, 0.3, args.epochs
    )

    adversaries = attacks_dispatcher(model=vanilla_classifier, dataset="cifarnorm")
    adversarial_apply = AdverApply(adversaries=adversaries)

    # --------------------------------------------------------------------------

    # WandB logging
    if args.wandb:
        wandb.init(
            project="carso-for-neurips-2023",
            config={
                "base_model": "cifar10_linf_wrn2810_a3_b10_t4_20m (Cui, 2023)",
                "batch_size": args.batchsize,
                "epochs": args.epochs,
                "loss_function": "Pixelwise Binary CrossEntropy; Reduction: Sum",
                "optimizer": "RAdam + Lookahead (5 steps)",
                "scheduler": f"CyclicLR (triangular): base_lr={adapted_lr_min}, max_lr={adapted_lr_max}, up_frac=0.3, total_steps={args.epochs}",
                "attacks": "FGSM Linf eps=4/255 eps=8/255 + PGD Linf eps=4/255 eps=8/255 steps=40 alpha=0.01",
                "batchwise_adversarial_fraction": args.advfrac,
            },
        )

    # --------------------------------------------------------------------------
    carso_machinery.train()
    carso_machinery.notify_train_eval_changes(armed=True, hardened=False)

    print("\nTraining...\n")

    for _ in trange(args.epochs, desc="Training epoch"):  # type: ignore
        # ----------------------------------------------------------------------
        for batch_idx, batched_datapoint in tqdm(  # type: ignore
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

            data, _, old_data = batched_datapoint

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

        scheduler.step()

    # --------------------------------------------------------------------------
    if args.save_model or args.wandb:
        model_namepath_compressor = (
            "../models/carso_reprcompressor_cuiwrn2810_cifar10_adv.pth"
        )
        model_namepath_dec = "../models/carso_dec_cuiwrn2810_cifar10_adv.pth"
        if isinstance(optimizer, Lookahead):
            optimizer._backup_and_load_cache()
        th.save(
            carso_machinery.repr_compressor.state_dict(),
            model_namepath_compressor,
        )
        th.save(carso_machinery.dec.state_dict(), model_namepath_dec)

    if args.wandb:
        repr_compressor = wandb.Artifact(
            "carso_reprcompressor_cuiwrn2810_cifar10_adv", type="model"
        )
        carso_dec = wandb.Artifact("carso_dec_cuiwrn2810_cifar10_adv", type="model")
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
