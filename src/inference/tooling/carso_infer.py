#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import List
from typing import Optional

import torch as th
from ebtorch.nn.utils import gather_model_repr
from torch import Tensor


def carso_ensembled_infer(  # pylint: disable=too-many-arguments
    classifier,
    pre_generative,
    generative,
    x: Tensor,
    latent_sample_size: int,
    named_layers: Optional[List[str]] = None,
    ensemble_size: int = 1000,
    device=None,
):

    # Handle device placement
    if device is None:
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
    x = x.to(device)
    classifier = classifier.eval()
    pre_generative = pre_generative.eval()
    generative = generative.eval()
    classifier = classifier.to(device)
    pre_generative = pre_generative.to(device)
    generative = generative.to(device)

    with th.no_grad():

        # Gather representations from classifier
        _, representations, _ = gather_model_repr(
            model=classifier,
            xin=x,
            device=device,
            named_layers=named_layers,
            preserve_graph=False,
        )
        del _

        representations = pre_generative(representations)

        # Collect votes
        ballot: list = []
        for _ in range(ensemble_size):
            voted_class = classifier(
                generative(
                    th.cat(
                        (
                            th.randn(
                                (representations.shape[0], latent_sample_size),
                                device=device,
                            ),
                            representations,
                        ),
                        dim=1,
                    )
                )
            ).argmax(dim=1, keepdim=True)
            ballot.append(voted_class)

        # Collate votes
        ret_batch, _ = th.mode(th.cat(ballot, dim=1).cpu())

        # Return
        return ret_batch.to(device)
