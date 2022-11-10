#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---- IMPORTS ----
import torch as th
from advertorch.attacks import DeepfoolLinfAttack
from advertorch.attacks import GradientSignAttack
from advertorch.attacks import LinfPGDAttack
from torchattacks.attacks.apgd import APGD

from .adapters.ta2at import TA2ATAdapter


random_noise_not_implemented: str = "Random noise attack not implemented yet"


def attacks_dispatcher(  # pylint: disable=too-many-arguments #NOSONAR
    model,
    fgsm: bool = True,
    pgd: bool = True,
    deepfool: bool = False,
    apgd_dlr: bool = False,
    randomnoise: bool = False,
    weak: bool = True,
    strong: bool = True,
    strongest: bool = False,
):
    adversaries = []

    if pgd and weak:
        adversaries.append(
            LinfPGDAttack(
                model,
                loss_fn=th.nn.CrossEntropyLoss(reduction="sum"),
                eps=0.15,
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
                eps=0.3,
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
                eps=0.5,
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
                eps=0.15,
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
                eps=0.3,
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
                eps=0.5,
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
                eps=0.15,
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
                eps=0.3,
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
                eps=0.5,
                clip_min=0.0,
                clip_max=1.0,
                targeted=False,
            )
        )

    if apgd_dlr and weak:
        adversaries.append(
            TA2ATAdapter(
                APGD(
                    model=model, norm="Linf", eps=0.15, steps=50, loss="dlr", rho=0.05
                ),
                xsize=(28, 28),
            )
        )

    if apgd_dlr and strong:
        adversaries.append(
            TA2ATAdapter(
                APGD(model=model, norm="Linf", eps=0.3, steps=50, loss="dlr", rho=0.05),
                xsize=(28, 28),
            )
        )

    if apgd_dlr and strongest:
        adversaries.append(
            TA2ATAdapter(
                APGD(model=model, norm="Linf", eps=0.5, steps=50, loss="dlr", rho=0.05),
                xsize=(28, 28),
            )
        )

    if randomnoise and weak:
        raise NotImplementedError(random_noise_not_implemented)

    if randomnoise and strong:
        raise NotImplementedError(random_noise_not_implemented)

    if randomnoise and strongest:
        raise NotImplementedError(random_noise_not_implemented)

    return adversaries
