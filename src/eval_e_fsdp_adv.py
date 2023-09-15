#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  Copyright (c) 2023 Emanuele Ballarin <emanuele@ballarin.cc>
#  Released under the terms of the MIT License
#  (see: https://url.ballarin.cc/mitlicense)
#
# ------------------------------------------------------------------------------
import argparse

import autoattack as aatk
import torch as th
import torch.distributed as dist
from carso import CARSOWrap
from ebtorch.data import cifarhundred_dataloader_dispatcher
from ebtorch.data import data_prep_dispatcher_3ch
from ebtorch.distributed import slurm_nccl_env
from ebtorch.nn import WideResNet
from torch.distributed.fsdp import CPUOffload
from torch.distributed.fsdp import FullyShardedDataParallel as FSDParallel
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy as auto_wrap_policy
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm


# ------------------------------------------------------------------------------
def main_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WideResNet-28-10+CARSO on CIFAR100 inference and comparison"
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=8 / 255,
        metavar="<epsilon>",
        help="Strength of the attack (default: 8/255)",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=800,
        metavar="<batch_size>",
        help="Batch size for testing (default: 800)",
    )
    parser.add_argument(
        "--ensemble_numerosity",
        type=int,
        default=8,
        metavar="<batch_size>",
        help="Size of the ensemble used to perform inference (default: 8)",
    )
    args = parser.parse_args()
    return args


# ------------------------------------------------------------------------------
def main_run(args: argparse.Namespace) -> None:
    # --------------------------------------------------------------------------
    # Distributed devices setup
    (
        rank,
        world_size,
        _,
        cpus_per_task,
        local_rank,
        device,
    ) = slurm_nccl_env()

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    th.cuda.set_device(device)
    # --------------------------------------------------------------------------

    # Dataset/DataLoader
    # Repeated twice just to allow gathering of dataset for DistributedSampler
    _, test_dl, _ = cifarhundred_dataloader_dispatcher(
        batch_size_train=1,
        batch_size_test=args.batchsize,
        cuda_accel=True,
        shuffle_test=False,
        unshuffle_train=True,
    )
    _, test_dl, _ = cifarhundred_dataloader_dispatcher(
        batch_size_train=1,
        batch_size_test=args.batchsize,
        cuda_accel=True,
        shuffle_test=False,
        unshuffle_train=True,
        dataloader_kwargs={
            "sampler": DistributedSampler(test_dl.dataset),
            "num_workers": cpus_per_task,
        },
    )
    del _

    # --------------------------------------------------------------------------
    adversarial_classifier = WideResNet(num_classes=100, bn_momentum=0.01)
    if not rank:
        adversarial_classifier.load_state_dict(
            th.load("../models/cifar100_a5_b12_t4_50m_w.pt")
        )
    adversarial_classifier.eval()

    carso_machinery = CARSOWrap(
        # Relevant
        wrapped_model=adversarial_classifier,
        input_data_height=32,
        input_data_width=32,
        input_data_channels=3,
        wrapped_repr_size=286820,
        compressed_repr_data_size=2816,
        shared_musigma_layer_size=192,
        sampled_code_size=128,
        ensemble_numerosity=args.ensemble_numerosity,
        input_data_no_compress=False,
        input_data_conv_flatten=True,
        repr_data_no_compress=False,
        slim_neck_repr_compressor=True,
        is_deconvolutional_decoder=True,
        is_cifar_decoder=10,
        binarize_repr=False,
        input_preprocessor=data_prep_dispatcher_3ch(device, post_flatten=False),
        differentiable_inference=True,
        sum_of_softmaxes_inference=True,
        suppress_stochastic_inference=True if args.ensemble_numerosity == 1 else False,
        output_logits=True,
        headless_mode=False,
        # Forced/Dummy
        compressed_input_data_size=0,
        convolutional_input_compressor=False,
    )

    if not rank:
        carso_machinery.repr_compressor.load_state_dict(
            th.load("../models/carso_reprcompressor_cuiwrn2810_cifar100_adv.pth")
        )
        carso_machinery.dec.load_state_dict(
            th.load("../models/carso_dec_cuiwrn2810_cifar100_adv.pth")
        )

    # --------------------------------------------------------------------------
    repr_layers = (
        "layer.1.block.0.conv_0",
        "layer.1.block.1.conv_1",
        "layer.2.block.0.conv_1",
        "layer.2.block.2.conv_1",
        "logits",
    )
    carso_machinery.set_repr_layers_names_lookup(repr_layers)
    # --------------------------------------------------------------------------

    carso_machinery = FSDParallel(
        module=carso_machinery,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        cpu_offload=CPUOffload(offload_params=False),
        limit_all_gathers=False,
        sync_module_states=True,
        use_orig_params=True,
        device_id=local_rank,
    )
    carso_machinery.eval()

    # --------------------------------------------------------------------------
    atk_dict_args = {
        "norm": "Linf",
        "eps": args.eps,
        "version": "rand",
        "verbose": False,
    }

    attack_adv_model = aatk.AutoAttack(carso_machinery, **atk_dict_args)

    # --------------------------------------------------------------------------
    # Evaluation counters
    number_of_elem_global_item: int = 0
    carso_correct_global_item: int = 0
    carso_adv_correct_global_item: int = 0

    test_dl.sampler.set_epoch(0)  # type: ignore

    for _, (true_data, true_label) in tqdm(  # type: ignore
        iterable=enumerate(test_dl),
        total=len(test_dl),
        desc="Testing batch",
        disable=(local_rank != 0),
    ):
        true_data, true_label = true_data.to(device), true_label.to(device)

        fake_data_adv = attack_adv_model.run_standard_evaluation(
            true_data,
            true_label,
            bs=args.batchsize,
        )

        adversarial_classifier = adversarial_classifier.eval()
        #
        # FIXME: Missing step -> disable sum-of-softmaxes infrence
        #        Before: we want to attack with BPDA (i.e. sum-of-softmaxes)
        #        After: we want to defend as designed (i.e. WITHOUT sum-of-softmaxes)
        #               but FSDP does not allow to access methods of wrapped model
        #               without re-instantiation.
        #       RESULT: this is a robust accuracy LOWER BOUND.
        #
        carso_machinery.eval()

        # ----------------------------------------------------------------------
        with th.no_grad():
            # ------------------------------------------------------------------
            carso_clean_class = carso_machinery(true_data)
            carso_pertu_class = carso_machinery(fake_data_adv)
            carso_clean_class, carso_pertu_class = carso_clean_class.argmax(
                dim=1, keepdim=True
            ).to(device), carso_pertu_class.argmax(dim=1, keepdim=True).to(device)
            # ------------------------------------------------------------------
            trueclass = true_label.flatten()
            carsoclass = carso_clean_class.flatten()
            carsoadv = carso_pertu_class.flatten()

            # Record results
            number_of_elem_global_item += true_data.shape[0]
            carso_correct_global_item += (
                th.eq(trueclass, carsoclass).count_nonzero().item()
            )
            carso_adv_correct_global_item += (
                th.eq(trueclass, carsoadv).count_nonzero().item()
            )
    # --------------------------------------------------------------------------
    carso_acc = carso_correct_global_item / number_of_elem_global_item
    carso_adv_acc = carso_adv_correct_global_item / number_of_elem_global_item
    # ------------------------------------------------------------------

    # Printout
    print("\n\n")
    print(f"CARSO ACCURACY                    : {carso_acc}")
    print("\n")
    print(f"CARSO ACCURACY UNDER ATTACK       : {carso_adv_acc}")
    print("\n\n")

    # --------------------------------------------------------------------------

    dist.destroy_process_group()


# ------------------------------------------------------------------------------


def main() -> None:
    parser_output = main_parse()
    main_run(parser_output)


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
