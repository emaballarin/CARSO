#!/usr/bin/env bash
#
#  Copyright (c) 2023 Emanuele Ballarin <emanuele@ballarin.cc>
#  Released under the terms of the MIT License
#  (see: https://url.ballarin.cc/mitlicense)
#
# ------------------------------------------------------------------------------

curl -O https://bucket.ballarin.cc/aimldl/carso2023/models/sc_a/fcn_mnist_clean.pth
curl -O https://bucket.ballarin.cc/aimldl/carso2023/models/sc_a/fcn_mnist_adv.pth
curl -O https://bucket.ballarin.cc/aimldl/carso2023/models/sc_a/carso_reprcompressor_fcn_mnist_adv.pth
curl -O https://bucket.ballarin.cc/aimldl/carso2023/models/sc_a/carso_dec_fcn_mnist_adv.pth
curl -O https://bucket.ballarin.cc/aimldl/carso2023/models/sc_a/cnn_mnist_clean.pth
curl -O https://bucket.ballarin.cc/aimldl/carso2023/models/sc_a/cnn_mnist_adv.pth
curl -O https://bucket.ballarin.cc/aimldl/carso2023/models/sc_a/carso_reprcompressor_cnn_mnist_adv.pth
curl -O https://bucket.ballarin.cc/aimldl/carso2023/models/sc_a/carso_dec_cnn_mnist_adv.pth
curl -O https://bucket.ballarin.cc/aimldl/carso2023/models/sc_a/cnn_fashionmnist_clean.pth
curl -O https://bucket.ballarin.cc/aimldl/carso2023/models/sc_a/cnn_fashionmnist_adv.pth
curl -O https://bucket.ballarin.cc/aimldl/carso2023/models/sc_a/carso_reprcompressor_cnn_fashionmnist_adv.pth
curl -O https://bucket.ballarin.cc/aimldl/carso2023/models/sc_a/carso_dec_cnn_fashionmnist_adv.pth
curl -O https://bucket.ballarin.cc/aimldl/carso2023/models/sc_b/carso_dec_wongrn18_cifar10_adv.pth
curl -O https://bucket.ballarin.cc/aimldl/carso2023/models/sc_b/carso_reprcompressor_wongrn18_cifar10_adv.pth
curl -O https://bucket.ballarin.cc/aimldl/carso2023/models/sc_b/cifar_model_weights_30_epochs.pth
curl -O https://bucket.ballarin.cc/aimldl/carso2023/models/sc_c/carso_dec_rebuffirn18_cifar10_adv.pth
curl -O https://bucket.ballarin.cc/aimldl/carso2023/models/sc_c/carso_reprcompressor_rebuffirn18_cifar10_adv.pth
curl -O https://bucket.ballarin.cc/aimldl/carso2023/models/sc_c/cifar10_linf_resnet18_ddpm.pt
