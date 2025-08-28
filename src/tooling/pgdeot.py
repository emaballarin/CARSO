#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
#  Copyright (c) 2023 Minjong Lee, Dongwoo Kim
#            (c) 2025 Emanuele Ballarin <emanuele@ballarin.cc>
#  Released under the terms of the MIT License
#  (see: https://url.ballarin.cc/mitlicense)
# ──────────────────────────────────────────────────────────────────────────────
from collections.abc import Callable
from typing import Tuple

import torch as th
import torch.nn.functional as F
from torch import Tensor


# ──────────────────────────────────────────────────────────────────────────────
class PGD:
    def __init__(
        self,
        get_logit: Callable[[Tensor], Tensor],
        attack_steps: int = 200,
        eps: float = 8.0 / 255.0,
        step_size: float = 0.007,
        eot: int = 20,
    ) -> None:
        self.clamp: Tuple[int, int] = (0, 1)
        self.eps: float = eps
        self.step_size: float = step_size
        self.get_logit: Callable[[Tensor], Tensor] = get_logit
        self.attack_steps: int = attack_steps
        self.eot: int = eot

    def _random_init(self, x: Tensor) -> Tensor:
        x: Tensor = x + (th.rand(x.size(), dtype=x.dtype, device=x.device) - 0.5) * 2 * self.eps
        return th.clamp(x, *self.clamp)

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        return self.forward(x, y)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x_adv: Tensor = x.detach().clone()
        for _ in range(self.attack_steps):
            grad: Tensor = th.zeros_like(x_adv)

            for _ in range(self.eot):
                x_adv.requires_grad = True

                logits: Tensor = self.get_logit(x_adv)

                loss: Tensor = F.cross_entropy(logits, y, reduction="sum")

                grad += th.autograd.grad(loss, [x_adv])[0].detach()
                x_adv: Tensor = x_adv.detach()

            grad /= self.eot
            grad: Tensor = grad.sign()
            x_adv: Tensor = x_adv + self.step_size * grad

            x_adv: Tensor = x + th.clamp(x_adv - x, min=-self.eps, max=self.eps)
            x_adv: Tensor = x_adv.detach()
            x_adv: Tensor = th.clamp(x_adv, *self.clamp)

        return x_adv
