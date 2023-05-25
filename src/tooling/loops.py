#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  Copyright (c) 2023 Emanuele Ballarin <emanuele@ballarin.cc>
#  Released under the terms of the MIT License
#  (see: https://url.ballarin.cc/mitlicense)
#
# ------------------------------------------------------------------------------
from typing import Optional

import torch as th
from ebtorch.logging import AverageMeter
from torch import nn as thnn
from torch.optim import Optimizer as thOptim
from tqdm.auto import tqdm
from tqdm.auto import trange


def train_epoch(
    model: thnn.Module,
    device: str,
    train_loader,
    loss_fn,
    optimizer: thOptim,
    epoch: int,
    print_every_nbatch: int,
    train_acc_avgmeter: AverageMeter,
    quiet: bool = False,
    adversaries: Optional[list] = None,
):
    if adversaries is None:
        adversaries: list = []

    model: thnn.Module = model.to(device)
    model: thnn.Module = model.train()

    train_acc_avgmeter.reset()
    for batch_idx, batched_datapoint in tqdm(
        enumerate(train_loader), total=len(train_loader), desc="Training batch"
    ):
        for adversary_idx in trange(
            len(adversaries) + 1, leave=False, desc="Adversary"
        ):
            data, target = batched_datapoint
            data, target = data.to(device), target.to(device)  # Before the attack

            if adversary_idx > 0:
                data = (
                    adversaries[adversary_idx - 1]
                    .perturb(data, target)
                    .reshape(data.shape)
                )

            data, target = data.to(device), target.to(device)  # After the attack

            optimizer.zero_grad()
            output: th.Tensor = model(data)
            loss = loss_fn(output, target)

            loss.backward()

            optimizer.step()

            train_acc_avgmeter.update(loss.item())

        if not quiet and batch_idx % print_every_nbatch == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100.0 * batch_idx / len(train_loader)}%)]\tAverage {train_acc_avgmeter.name}: {train_acc_avgmeter.avg}"
            )


def test(
    model: thnn.Module,
    device,
    test_loader,
    loss_fn,
    test_acc_avgmeter: AverageMeter,
    quiet: bool = False,
):
    test_loss: int = 0
    correct: int = 0

    model: thnn.Module = model.to(device)
    model: thnn.Module = model.eval()

    test_acc_avgmeter.reset()
    with th.no_grad():
        for data, target in tqdm(
            iterable=test_loader, desc="Testing batch", leave=False
        ):
            data, target = data.to(device), target.to(device)

            output = model(data)

            test_loss += loss_fn(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max (log-)probability class
            correct += pred.eq(target.view_as(pred)).sum().item()

    ltlds: int = len(test_loader.dataset)
    test_loss /= ltlds

    if not quiet:
        print(
            f"Average loss: {test_loss}, Accuracy: {correct}/{ltlds} ({100.0 * correct / ltlds}%)"
        )

    return test_loss, correct / ltlds
