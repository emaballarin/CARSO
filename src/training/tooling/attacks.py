#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---- IMPORTS ----
import torch as th
from advertorch.attacks import DeepfoolLinfAttack
from advertorch.attacks import GradientSignAttack
from advertorch.attacks import LinfPGDAttack


def attacks_dispatcher(
    model,
    fgsm: bool = True,
    pgd: bool = True,
    deepfool: bool = False,
    weak: bool = True,
    strong: bool = True,
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

    return adversaries
