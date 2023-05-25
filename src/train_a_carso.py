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
from ebtorch.nn import beta_reco_bce
from ebtorch.nn.utils import AdverApply
from ebtorch.optim import ralah_optim
from tooling.architectures import fashionmnist_cnn_classifier_dispatcher
from tooling.architectures import mnist_cnn_classifier_dispatcher
from tooling.architectures import mnist_data_prep_dispatcher
from tooling.architectures import mnist_fcn_classifier_dispatcher
from tooling.attacks import attacks_dispatcher
from tooling.data import fashionmnist_dataloader_dispatcher
from tooling.data import mnist_dataloader_dispatcher
from torch.optim.lr_scheduler import CyclicLR
from tqdm.auto import tqdm
from tqdm.auto import trange

# ------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FCN/CNN+CARSO on MNIST/FashionMNIST training"
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
    parser.add_argument(
        "--base_model_type",
        type=str,
        default="fcn",
        metavar="<base_model_type>",
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

    train_dl, _, _ = data_dispatcher(
        batch_size_train=args.batchsize,
        batch_size_test=1,
        cuda_accel=device == th.device("cuda"),
    )
    del _

    # --------------------------------------------------------------------------

    # Model selection
    if args.base_model_type == "fcn":
        classifier_model = mnist_fcn_classifier_dispatcher(kwta_filter=False)
    elif args.base_model_type == "cnn":
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

    classifier_model.load_state_dict(
        th.load(f"../models/{args.base_model_type}_{args.dataset}_adv.pth")
    )
    classifier_model.to(device).eval()

    # --------------------------------------------------------------------------
    carso_machinery = CARSOWrap(
        # Relevant
        wrapped_model=classifier_model,
        input_data_height=28,
        input_data_width=28,
        input_data_channels=1,
        wrapped_repr_size=290 if args.base_model_type == "fcn" else 4814,
        compressed_repr_data_size=130,
        shared_musigma_layer_size=96,
        sampled_code_size=64,
        input_data_no_compress=False,
        input_data_conv_flatten=True,
        repr_data_no_compress=False,
        slim_neck_repr_compressor=True,
        is_deconvolutional_decoder=True,
        is_cifar_decoder=False,
        binarize_repr=False,
        input_preprocessor=(
            mnist_data_prep_dispatcher(post_flatten=False)
            if args.dataset == "mnist"
            else th.nn.Identity()
        ),
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

    # --------------------------------------------------------------------------
    # Representation layers selection
    if args.base_model_type == "fcn":
        repr_layers = (
            "2.module_battery.1",
            "2.module_battery.5",
            "2.module_battery.9",
        )
    elif args.base_model_type == "cnn":
        repr_layers = ("1", "4", "9", "13")
    else:
        raise ValueError("Invalid model selected! Valid options: fcn, cnn")

    # Adapt learning rate to batch size (heuristically: linear scaling)
    lr_magic_constant: float = 3.0
    adapted_lr_max: float = lr_magic_constant * 1e-5 * args.batchsize
    adapted_lr_min: float = 0.5e-8

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

    adversaries = attacks_dispatcher(model=classifier_model, dataset="xnist")
    adversarial_apply = AdverApply(adversaries=adversaries)

    # --------------------------------------------------------------------------

    # WandB logging
    if args.wandb:
        wandb.init(
            project="carso-for-neurips-2023",
            config={
                "base_model": f"Custom-trained {args.base_model_type.upper()} on {args.dataset.upper()}",
                "batch_size": args.batchsize,
                "epochs": args.epochs,
                "loss_function": "Pixelwise Binary CrossEntropy; Reduction: Sum",
                "optimizer": "RAdam + Lookahead (5 steps)",
                "scheduler": f"CyclicLR (triangular): base_lr={adapted_lr_min}, max_lr={adapted_lr_max}, epochs_up={int(1 * args.epochs / 3)}, epochs_down={int(args.epochs - int(1 * args.epochs / 3))}",
                "attacks": "FGSM Linf eps=4/255 eps=8/255 + PGD Linf eps=4/255 eps=8/255 steps=40 alpha=0.01",
                "batchwise_adversarial_fraction": args.advfrac,
            },
        )

    # --------------------------------------------------------------------------
    carso_machinery.train()
    carso_machinery.notify_train_eval_changes(armed=True, hardened=False)
    # --------------------------------------------------------------------------

    print("\nTraining...\n")

    for epoch in trange(args.epochs, desc="Training epoch"):
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

        scheduler.step()

    # --------------------------------------------------------------------------
    if args.save_model or args.wandb:
        model_namepath_compressor = f"../models/carso_reprcompressor_{args.base_model_type}_{args.dataset}_adv.pth"
        model_namepath_dec = (
            f"../models/carso_dec_{args.base_model_type}_{args.dataset}_adv.pth"
        )

        optimizer._backup_and_load_cache()
        th.save(
            carso_machinery.repr_compressor.state_dict(),
            model_namepath_compressor,
        )
        th.save(carso_machinery.dec.state_dict(), model_namepath_dec)

    if args.wandb:
        repr_compressor = wandb.Artifact(
            f"carso_reprcompressor_{args.base_model_type}_{args.dataset}_adv",
            type="model",
        )
        carso_dec = wandb.Artifact(
            f"carso_dec_{args.base_model_type}_{args.dataset}_adv", type="model"
        )
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
