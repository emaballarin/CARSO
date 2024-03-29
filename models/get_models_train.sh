#!/usr/bin/env bash
#
#  Copyright (c) 2023 Emanuele Ballarin <emanuele@ballarin.cc>
#  Released under the terms of the MIT License
#  (see: https://url.ballarin.cc/mitlicense)
#
# ------------------------------------------------------------------------------

curl -O https://bucket.ballarin.cc/aimldl/carso2023/models/sc_b/cifar_model_weights_30_epochs.pth
curl -O https://bucket.ballarin.cc/aimldl/carso2023/models/sc_c/cifar10_linf_resnet18_ddpm.pt
curl -O https://bucket.ballarin.cc/aimldl/carso2023/models/sc_d/cifar10_a3_b10_t4_20m_w.pt
curl -O https://bucket.ballarin.cc/aimldl/carso2023/models/sc_e/cifar100_a5_b12_t4_50m_w.pt
