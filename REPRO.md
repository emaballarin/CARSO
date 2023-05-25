# `CARSO`: *Counter-Adversarial Recall of Synthetic Observations*

## Reproducibility requirements

The code has been tested with `Python 3.10.8`, running on `Ubuntu 20.04`. The only system-wide requirements are a `bash` shell and availability of the `curl` command. A working *base* Python install is also required. Additional dependencies are listed in the `requirements.txt` file, and may be installed by `pip install -r ./requirements.txt`. The code is also compatible with `Conda`-based installs, provided the same requirements are satisfied.

In order to fruitfully use *GPU* acceleration, working *kernel-space* *NVIDIA CUDA*-compatible drivers are required (the exact version depending on *PyTorch* compatibility and the specific system in use).

## Datasets, models, *pre-trained* models

The datasets employed (*MNIST*, *FashionMNIST*, *CIFAR-10*) are downloaded automatically by the scripts provided. A working Internet connection is required for their download.

The code requires pre-trained models from *[Wong at al., 2020]* and *[Rebuffi et al., 2021]*, obtainable from an `s3` storage bucket (set up *ad hoc* for the sake of reproducibility), by `cd ./models/; bash get_models_train.sh`. Such setup allows reproduction of the training phases from which the experimental results of the paper are obtained.

To further download the entire set of *pre-trained* models required to reproduce the evaluation phase only (without actually performing the training itself), `cd ./models/; bash get_models_test.sh` is required. Pre-trained models will be downloaded from the same `s3` storage bucket.

## Model training

To reproduce the training phase in its entirety, the following commands are to be issued.

```bash
# Scenario A, Clean training
python -O train_a_basemodel.py --save_model --model_type fcn --dataset mnist
python -O train_a_basemodel.py --save_model --model_type cnn --dataset mnist
python -O train_a_basemodel.py --save_model --model_type cnn --dataset fashionmnist

# Scenario A, Adversarial training
python -O train_a_basemodel.py --save_model --model_type fcn --dataset mnist --do_attack
python -O train_a_basemodel.py --save_model --model_type cnn --dataset mnist --do_attack
python -O train_a_basemodel.py --save_model --model_type cnn --dataset fashionmnist --do_attack

# Scenario A, CARSO training
python -O train_a_carso.py --save_model --base_model_type fcn --dataset mnist
python -O train_a_carso.py --save_model --base_model_type cnn --dataset mnist
python -O train_a_carso.py --save_model --base_model_type cnn --dataset fashionmnist

# Scenario B, CARSO training
python -O train_b.py --save_model

# Scenario C, CARSO training
python -O train_c.py --save_model
```

The `python` command is assumed to be the one from the environment set up earlier. The *scenarios* referred to mirror the description provided within the paper.

## Model evaluation

To reproduce the testing phase in its entirety, the following commands are to be issued.

```bash
# Scenario A: individual attacks
python -O eval_a_individual.py --attack pgd --strength s --base_model_type fcn --dataset mnist
python -O eval_a_individual.py --attack pgd --strength x --base_model_type fcn --dataset mnist
python -O eval_a_individual.py --attack fgs --strength s --base_model_type fcn --dataset mnist
python -O eval_a_individual.py --attack fgs --strength x --base_model_type fcn --dataset mnist
python -O eval_a_individual.py --attack dfl --strength w --base_model_type fcn --dataset mnist
python -O eval_a_individual.py --attack dfl --strength s --base_model_type fcn --dataset mnist
python -O eval_a_individual.py --attack dfl --strength x --base_model_type fcn --dataset mnist
python -O eval_a_individual.py --attack apg --strength w --base_model_type fcn --dataset mnist
python -O eval_a_individual.py --attack apg --strength s --base_model_type fcn --dataset mnist
python -O eval_a_individual.py --attack apg --strength x --base_model_type fcn --dataset mnist

python -O eval_a_individual.py --attack pgd --strength s --base_model_type cnn --dataset mnist
python -O eval_a_individual.py --attack pgd --strength x --base_model_type cnn --dataset mnist
python -O eval_a_individual.py --attack fgs --strength s --base_model_type cnn --dataset mnist
python -O eval_a_individual.py --attack fgs --strength x --base_model_type cnn --dataset mnist
python -O eval_a_individual.py --attack dfl --strength w --base_model_type cnn --dataset mnist
python -O eval_a_individual.py --attack dfl --strength s --base_model_type cnn --dataset mnist
python -O eval_a_individual.py --attack dfl --strength x --base_model_type cnn --dataset mnist
python -O eval_a_individual.py --attack apg --strength w --base_model_type cnn --dataset mnist
python -O eval_a_individual.py --attack apg --strength s --base_model_type cnn --dataset mnist
python -O eval_a_individual.py --attack apg --strength x --base_model_type cnn --dataset mnist

python -O eval_a_individual.py --attack pgd --strength s --base_model_type cnn --dataset fashionmnist
python -O eval_a_individual.py --attack pgd --strength x --base_model_type cnn --dataset fashionmnist
python -O eval_a_individual.py --attack fgs --strength s --base_model_type cnn --dataset fashionmnist
python -O eval_a_individual.py --attack fgs --strength x --base_model_type cnn --dataset fashionmnist
python -O eval_a_individual.py --attack dfl --strength w --base_model_type cnn --dataset fashionmnist
python -O eval_a_individual.py --attack dfl --strength s --base_model_type cnn --dataset fashionmnist
python -O eval_a_individual.py --attack dfl --strength x --base_model_type cnn --dataset fashionmnist
python -O eval_a_individual.py --attack apg --strength w --base_model_type cnn --dataset fashionmnist
python -O eval_a_individual.py --attack apg --strength s --base_model_type cnn --dataset fashionmnist
python -O eval_a_individual.py --attack apg --strength x --base_model_type cnn --dataset fashionmnist


# Scenario B: individual attacks
python -O eval_b_individual.py --attack pgd --strength s
python -O eval_b_individual.py --attack pgd --strength x
python -O eval_b_individual.py --attack fgs --strength s
python -O eval_b_individual.py --attack fgs --strength x
python -O eval_b_individual.py --attack dfl --strength w
python -O eval_b_individual.py --attack dfl --strength s
python -O eval_b_individual.py --attack dfl --strength x
python -O eval_b_individual.py --attack apg --strength w
python -O eval_b_individual.py --attack apg --strength s
python -O eval_b_individual.py --attack apg --strength x

python -O eval_b_individual.py --attack pgd --strength s --e2e
python -O eval_b_individual.py --attack pgd --strength x --e2e
python -O eval_b_individual.py --attack fgs --strength s --e2e
python -O eval_b_individual.py --attack fgs --strength x --e2e
python -O eval_b_individual.py --attack apg --strength w --e2e
python -O eval_b_individual.py --attack apg --strength s --e2e
python -O eval_b_individual.py --attack apg --strength x --e2e

# Scenario A: AutoAttack
python -O eval_a.py --base_model_type fcn --dataset mnist
python -O eval_a.py --base_model_type cnn --dataset mnist
python -O eval_a.py --base_model_type cnn --dataset fashionmnist

# Scenario B: AutoAttack
python -O eval_b.py
python -O eval_b.py --e2e --explicitly_random

# Scenario C: AutoAttack
python -O eval_c.py
python -O eval_c.py --e2e --explicitly_random

# Scenario C: AutoAttack against the "naked" extracted internal representation
python -O eval_b.py --e2e --noextract --explicitly_random
```

The `python` command is assumed to be the one from the environment set up earlier. The *scenarios* and *attacks* referred to mirror the description provided within the paper.

## Additional notes

The execution of the *training* and *evaluation* phases described above requires at least $30$ GiB of main memory (in the case of *CPU* processing) or dedicated graphical memory (in case of *GPU* acceleration). The execution of the commands above may require a long runtime, in dependence from the specific hardware setup and software versions.  On an updated, modern system equipped with an *NVIDIA A100 GPU* the execution of the entire list of commands (*i.e.* required for the reproduction of the training and evaluation phases) should not require more than $24$ hours. The evaluation phase is the most time-consuming.
