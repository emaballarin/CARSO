#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---- IMPORTS ----
import argparse

import torch as th
from ebtorch.logging import AverageMeter
from ebtorch.nn import GaussianReparameterizerSampler
from ebtorch.nn.utils import gather_model_repr
from ebtorch.nn.utils import model_reqgrad_
from ebtorch.optim import RAdam
from tooling.architectures import compressor_dispatcher
from tooling.architectures import fcn_carso_dispatcher
from tooling.architectures import mnist_data_prep_dispatcher
from tooling.architectures import mnistfcn_dispatcher
from tooling.architectures import pixelwise_bce_sum
from tooling.attacks import attacks_dispatcher
from tooling.data import mnist_dataloader_dispatcher
from torch import Tensor
from torch.optim.lr_scheduler import MultiStepLR


def main():  # NOSONAR # pylint: disable=too-many-locals,too-many-statements
    # Argument parsing...
    parser = argparse.ArgumentParser(description="FCN+CARSO on MNIST training")
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

    # ---- TRAINING TUNING ----
    TRAIN_BATCHSIZE: int = 128
    TEST_BATCHSIZE: int = 512
    TRAIN_EPOCHS: int = 60
    PRINT_EVERY_NEP = 120

    # ---- DATASETS ----
    train_dl, _, _ = mnist_dataloader_dispatcher(
        batch_size_train=TRAIN_BATCHSIZE,
        batch_size_test=TEST_BATCHSIZE,
        cuda_accel=bool(device == th.device("cuda")),
    )
    del _

    vanilla_classifier = mnistfcn_dispatcher()
    vanilla_classifier.load_state_dict(th.load("../../models/mnistfcn_adv.pth"))

    mnist_data_prep = mnist_data_prep_dispatcher()
    input_funnel = compressor_dispatcher(28 * 28, 28 * 28 // 4)
    repr_funnel = compressor_dispatcher(290, 290 // 5)

    carso_enc_neck, carso_enc_mu, carso_enc_sigma, carso_dec = fcn_carso_dispatcher(
        28 * 28 // 4,
        290 // 5,
        (28 * 28 // 4 + 290 // 5 + 36) // 2,
        36,
        actual_output_size=28 * 28,
        device=device,
    )

    gauss_rp_sampler = GaussianReparameterizerSampler()

    all_models = [
        vanilla_classifier,
        input_funnel,
        repr_funnel,
        carso_enc_neck,
        carso_enc_mu,
        carso_enc_sigma,
        carso_dec,
        gauss_rp_sampler,
        mnist_data_prep,
    ]

    for model in all_models:
        model = model.to(device)
        model = model.train()

    # Handle pretrained model(s)
    vanilla_classifier = vanilla_classifier.eval()
    model_reqgrad_(vanilla_classifier, set_to=False)

    # ---- OPTIMIZER AND SCHEDULER ----
    OPTIMIZER = RAdam(
        list(mnist_data_prep.parameters())
        + list(input_funnel.parameters())
        + list(repr_funnel.parameters())
        + list(carso_enc_neck.parameters())
        + list(carso_enc_mu.parameters())
        + list(carso_enc_sigma.parameters())
        + list(carso_dec.parameters())
        + list(gauss_rp_sampler.parameters()),
        lr=0.001,
    )

    SCHEDULER = MultiStepLR(OPTIMIZER, milestones=[25, 40, 50], gamma=0.7)
    INNER_SCHEDULER = None

    # ---- ADVERSARY ----
    if args.attack:
        adversaries = attacks_dispatcher(model=vanilla_classifier)
        namepiece: str = "adv"
    else:
        adversaries = []
        namepiece: str = "clean"

    # ---- TRAINING STATISTICS ----
    train_acc_avgmeter = AverageMeter("batchwise training loss")

    # ---- TRAINING LOOP (FULL) ----
    for epoch in range(1, TRAIN_EPOCHS + 1):

        # Preparation
        train_acc_avgmeter.reset()
        for model in all_models:
            model = model.train()
        vanilla_classifier = vanilla_classifier.eval()
        model_reqgrad_(vanilla_classifier, set_to=False)

        print(f"\n---- EPOCH {epoch} OF {TRAIN_EPOCHS} ----\n")

        # Inner loop
        for batch_idx, batched_datapoint in enumerate(train_dl):

            data, target = batched_datapoint
            data, target = data.to(device), target.to(device)  # Before the attack

            old_data = data.detach()  # Copy unperturbed input

            # Attack loop
            for adversary_idx in range(len(adversaries) + 1):

                if adversary_idx > 0:
                    data = (
                        adversaries[adversary_idx - 1]
                        .perturb(data.flatten(start_dim=1), target)
                        .reshape(data.shape)
                    )

                data, target = data.to(device), target.to(device)  # After the attack

            # Extract representation
            with th.no_grad():
                _, class_repr, _ = gather_model_repr(
                    vanilla_classifier,
                    data,
                    device,
                    ["2.module_battery.1", "2.module_battery.5", "2.module_battery.9"],
                    preserve_graph=False,
                )
                del _

            # Prepare for the next stage
            data.requires_grad_(True)
            class_repr.requires_grad_(True)

            # Record gradients from here
            OPTIMIZER.zero_grad()

            # Input encoding
            compress_input: Tensor = input_funnel(mnist_data_prep(data))

            # Representation encoding
            compress_repr: Tensor = repr_funnel(class_repr)

            # Training loop for CVAE

            # ENC
            cvae_input = th.cat((compress_input, compress_repr), dim=1)
            cvae_neck_out = carso_enc_neck(cvae_input)
            cvae_mu, cvae_sigma = (
                carso_enc_mu(cvae_neck_out),
                carso_enc_sigma(cvae_neck_out),
            )

            cvae_enc = gauss_rp_sampler(cvae_mu, cvae_sigma)

            # DEC
            cvae_dec_input = th.cat((cvae_enc, compress_repr), dim=1)
            input_reco = carso_dec(cvae_dec_input)

            # Compute loss
            kldiv = (0.5 * (cvae_mu**2 + th.exp(cvae_sigma) - cvae_sigma - 1)).sum()
            pwbce = pixelwise_bce_sum(input_reco, old_data.flatten(start_dim=1))
            loss = kldiv + pwbce
            loss.backward()

            # Optimize
            OPTIMIZER.step()
            if INNER_SCHEDULER is not None:  # NOSONAR
                INNER_SCHEDULER.step()

            # Track stats
            train_acc_avgmeter.update(loss.item())

            # Print
            if not args.quiet and batch_idx % PRINT_EVERY_NEP == 0:
                print(
                    f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_dl.dataset)} ({100.0 * batch_idx / len(train_dl)}%)]\tAverage {train_acc_avgmeter.avg}: {loss.item()}"
                )

    # Out of epoch
    SCHEDULER.step()

    if args.save_model:
        th.save(
            repr_funnel.state_dict(), "../../models/repr_funnel_" + namepiece + ".pth"
        )
        th.save(carso_dec.state_dict(), "../../models/carso_dec_" + namepiece + ".pth")


# Run!
if __name__ == "__main__":
    main()
