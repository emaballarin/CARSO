#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Any
from typing import Optional

import torch as th


def train_epoch(
    model,
    device,
    train_loader,
    loss_fn,
    optimizer,
    epoch,
    print_every_nep,
    train_acc_avgmeter,
    inner_scheduler: Optional[Any] = None,
    quiet: bool = False,
    adversaries: Optional[list] = None,
):

    if adversaries is None:
        adversaries: list = []

    model = model.to(device)
    model = model.train()

    train_acc_avgmeter.reset()
    for batch_idx, batched_datapoint in enumerate(train_loader):

        for adversary_idx in range(len(adversaries) + 1):

            data, target = batched_datapoint
            data, target = data.to(device), target.to(device)  # Before the attack

            if adversary_idx > 0:
                data = (
                    adversaries[adversary_idx - 1]
                    .perturb(data.flatten(start_dim=1), target)
                    .reshape(data.shape)
                )

            data, target = data.to(device), target.to(device)  # After the attack

            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)

            loss.backward()

            optimizer.step()
            if inner_scheduler is not None:
                inner_scheduler.step()

            train_acc_avgmeter.update(loss.item())

        if not quiet and batch_idx % print_every_nep == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100.0 * batch_idx / len(train_loader)}%)]\tAverage {train_acc_avgmeter.name}: {train_acc_avgmeter.avg}"
            )


def test(
    model,
    device,
    test_loader,
    loss_fn,
    test_acc_avgmeter,
    quiet: bool = False,
):

    test_loss: int = 0
    correct: int = 0

    model = model.to(device)
    model = model.eval()

    test_acc_avgmeter.reset()
    with th.no_grad():

        for batched_datapoint in test_loader:
            data, target = batched_datapoint
            data, target = data.to(device), target.to(device)

            output = model(data)

            test_loss += loss_fn(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max (log-)probability class
            correct += pred.eq(target.view_as(pred)).sum().item()

    ltlds = len(test_loader.dataset)
    test_loss /= ltlds

    if not quiet:
        print(
            f"Average loss: {test_loss}, Accuracy: {correct}/{ltlds} ({100.0 * correct / ltlds}%)"
        )

    return test_loss, correct / ltlds
