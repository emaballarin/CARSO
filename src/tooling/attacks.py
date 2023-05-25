#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  Copyright (c) 2023 Emanuele Ballarin <emanuele@ballarin.cc>
#  Released under the terms of the MIT License
#  (see: https://url.ballarin.cc/mitlicense)
#
# ------------------------------------------------------------------------------
from typing import List
from typing import Union

import torch as th
import torch.nn as thnn
from advertorch.attacks import Attack as ATAttack
from advertorch.attacks import DeepfoolLinfAttack
from advertorch.attacks import GradientSignAttack
from advertorch.attacks import LinfPGDAttack
from ebtorch.nn.utils import TA2ATAdapter
from torchattacks.attacks.apgd import APGD
from torchattacks.attacks.gn import GN


random_noise_not_implemented: str = "Random noise attack not implemented yet"


def attacks_dispatcher(  # pylint: disable=too-many-arguments #NOSONAR
    model: thnn.Module,
    fgsm: bool = True,
    pgd: bool = True,
    deepfool: bool = False,
    apgd_dlr: bool = False,
    randomnoise: bool = False,
    weak: bool = True,
    strong: bool = True,
    strongest: bool = False,
    dataset: str = "xnist",
    apgd_stochastic: bool = False,
):
    if dataset == "xnist":
        strengths: dict = {"w": 0.15, "s": 0.3, "x": 0.45}
    elif dataset == "cifarx":
        strengths: dict = {"w": 4 / 255, "s": 8 / 255, "x": 12 / 255}
    elif dataset == "cifarnorm":
        strengths: dict = {"w": 16 / 255, "s": 32 / 255, "x": 48 / 255}
    else:
        raise NotImplementedError("Dataset not supported... yet!")

    if apgd_stochastic:
        apgd_loss: str = "dlr"
        apgd_eot_iter: int = 10
    else:
        apgd_loss: str = "ce"
        apgd_eot_iter: int = 1

    adversaries: List[Union[ATAttack, TA2ATAdapter]] = []

    if pgd and weak:
        adversaries.append(
            LinfPGDAttack(
                model,
                loss_fn=th.nn.CrossEntropyLoss(reduction="sum"),
                eps=strengths["w"],
                nb_iter=40,
                eps_iter=0.01,
                rand_init=True,
                clip_min=0.0,
                clip_max=1.0,
                targeted=False,
            )
        )

    if pgd and strong:
        adversaries.append(
            LinfPGDAttack(
                model,
                loss_fn=th.nn.CrossEntropyLoss(reduction="sum"),
                eps=strengths["s"],
                nb_iter=40,
                eps_iter=0.01,
                rand_init=True,
                clip_min=0.0,
                clip_max=1.0,
                targeted=False,
            )
        )

    if pgd and strongest:
        adversaries.append(
            LinfPGDAttack(
                model,
                loss_fn=th.nn.CrossEntropyLoss(reduction="sum"),
                eps=strengths["x"],
                nb_iter=40,
                eps_iter=0.01,
                rand_init=True,
                clip_min=0.0,
                clip_max=1.0,
                targeted=False,
            )
        )

    if fgsm and weak:
        adversaries.append(
            GradientSignAttack(
                model,
                loss_fn=th.nn.CrossEntropyLoss(reduction="sum"),
                eps=strengths["w"],
                clip_min=0.0,
                clip_max=1.0,
                targeted=False,
            )
        )

    if fgsm and strong:
        adversaries.append(
            GradientSignAttack(
                model,
                loss_fn=th.nn.CrossEntropyLoss(reduction="sum"),
                eps=strengths["s"],
                clip_min=0.0,
                clip_max=1.0,
                targeted=False,
            )
        )

    if fgsm and strongest:
        adversaries.append(
            GradientSignAttack(
                model,
                loss_fn=th.nn.CrossEntropyLoss(reduction="sum"),
                eps=strengths["x"],
                clip_min=0.0,
                clip_max=1.0,
                targeted=False,
            )
        )

    if deepfool and weak:
        adversaries.append(
            DeepfoolLinfAttack(
                model,
                loss_fn=th.nn.CrossEntropyLoss(reduction="sum"),
                eps=strengths["w"],
                clip_min=0.0,
                clip_max=1.0,
                targeted=False,
            )
        )

    if deepfool and strong:
        adversaries.append(
            DeepfoolLinfAttack(
                model,
                loss_fn=th.nn.CrossEntropyLoss(reduction="sum"),
                eps=strengths["s"],
                clip_min=0.0,
                clip_max=1.0,
                targeted=False,
            )
        )

    if deepfool and strongest:
        adversaries.append(
            DeepfoolLinfAttack(
                model,
                loss_fn=th.nn.CrossEntropyLoss(reduction="sum"),
                eps=strengths["x"],
                clip_min=0.0,
                clip_max=1.0,
                targeted=False,
            )
        )

    if apgd_dlr and weak:
        adversaries.append(
            TA2ATAdapter(
                APGD(
                    model=model,
                    norm="Linf",
                    eps=strengths["w"],
                    steps=50,
                    loss=apgd_loss,
                    rho=0.05,
                    eot_iter=apgd_eot_iter,
                )
            )
        )

    if apgd_dlr and strong:
        adversaries.append(
            TA2ATAdapter(
                APGD(
                    model=model,
                    norm="Linf",
                    eps=strengths["s"],
                    steps=50,
                    loss=apgd_loss,
                    rho=0.05,
                    eot_iter=apgd_eot_iter,
                )
            )
        )

    if apgd_dlr and strongest:
        adversaries.append(
            TA2ATAdapter(
                APGD(
                    model=model,
                    norm="Linf",
                    eps=strengths["x"],
                    steps=50,
                    loss=apgd_loss,
                    rho=0.05,
                    eot_iter=apgd_eot_iter,
                )
            )
        )

    if randomnoise and weak:
        adversaries.append(TA2ATAdapter(GN(model=model, std=strengths["w"] / 2)))

    if randomnoise and strong:
        adversaries.append(TA2ATAdapter(GN(model=model, std=strengths["s"] / 2)))

    if randomnoise and strongest:
        adversaries.append(TA2ATAdapter(GN(model=model, std=strengths["x"] / 2)))

    return adversaries
