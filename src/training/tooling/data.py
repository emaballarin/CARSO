#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from pathlib import Path as pathlibPath
from typing import Optional
from typing import Union

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor


def mnist_dataloader_dispatcher(
    data_root: str = "../datasets/",
    batch_size_train: int = 256,
    batch_size_test: int = 512,
    cuda_accel: bool = False,
    dataset_kwargs: Optional[dict] = None,
    dataloader_kwargs: Optional[dict] = None,
):
    os.makedirs(name=data_root, exist_ok=True)

    transforms = Compose([ToTensor()])

    # Address dictionary mutability as default argument
    if dataset_kwargs is None:
        dataset_kwargs = {}
    if dataloader_kwargs is None:
        dataloader_kwargs = {}

    trainset = MNIST(
        root=data_root,
        train=True,
        transform=transforms,
        download=True,
        **dataset_kwargs
    )
    testset = MNIST(
        root=data_root,
        train=False,
        transform=transforms,
        download=True,
        **dataset_kwargs
    )

    cuda_args = {}
    if cuda_accel:
        cuda_args = {"num_workers": 1, "pin_memory": True}

    trainloader = DataLoader(
        trainset,
        batch_size=batch_size_train,
        shuffle=True,
        **cuda_args,
        **dataloader_kwargs
    )
    testloader = DataLoader(
        testset,
        batch_size=batch_size_test,
        shuffle=False,
        **cuda_args,
        **dataloader_kwargs
    )
    test_on_train_loader = DataLoader(
        trainset,
        batch_size=batch_size_test,
        shuffle=False,
        **cuda_args,
        **dataloader_kwargs
    )

    return trainloader, testloader, test_on_train_loader
