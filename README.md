# README

This repository contains the source code to reproduce the experiments of the paper "Deep Double Descent via Smooth Interpolation", M. Gamba, E. Englesson, M. Björkman, H. Azizpour. TMLR 2023.

If you use or draw inspiration from this code in your experiments, please cite as:

```latex
@article{
    gamba2023deep,
    title={Deep Double Descent via Smooth Interpolation},
    author={Matteo Gamba and Erik Englesson and M\{r}arten Bj{\"o}rkman and Hossein Azizpour},
    journal={Transactions on Machine Learning Research},
    year={2023},
    url={https://openreview.net/forum?id=fempQstMbV},
    note={}
}
```

## Set up

To install all required dependencies, run:
```bash
pip install -r requirements.txt
```

## Configure environment

To configure the computing environment for running the experiments, you can set the following environment variables
```bash
SAVE_DIR="/path/to/savedir" # basedir where to store results
NAME="exp_name" # name of the experiment run
DATA_DIR="/path/to/dataset/dir" # path where to load data from
DEVICE="cuda" # "cpu" or "cuda"
WORKERS=1 # number of CPU worker processes
```
or use the following command line arguments with `train.py`, `compute_stats.py` and `aggregate_stats.py`:
```bash
 --save-dir="/path/to/savedir"
 --name="exp_name"
 --data-dir="/path/to/dataset/dir"
 --device="cuda"
 --workers="1"
```

# Train a model

To train a model, run
```bash
python train.py --device=cuda --name='test' --save-dir='checkpoints' --data-dir="./data" --workers=4 --data="cifar10" --model="resnet18_64" --epochs=300 --batch-size=128 --augmentation --seed=42 --train-split=49000 --val-split=1000 --eval-every=10 --optimizer=adam --learning-rate=1e-4
```
model checkpoints for the corresponding run will be compressed to a single zip archive, and stored to `./checkpoints/test/resnet18_64/cifar10/augmentation/seed-42/checkpoints.zip`

# Compute statistics

To compute data-driven measures, the user must specify a metric to compute, a data generation strategy and what model checkpoint to load. A model checkpoint is identified by specifying a network architecture, dataset, training seed, checkpoint and dataset split:
```bash
python compute_stats.py --device="cuda" --name=NAME --save-dir=SAVE_DIR --workers=4 --data-dir=DATA_DIR --data=cifar10 --model=resnet18_64 --augmentation --seed 42 --checkpoints 1 --train-split=49000 --val-split=1000 --gen-strategy="1px-shifts" --normalization crossentropy --num-samples=49000 --batch-size=140 --num-directions=4 --metric=jacobian
```
Results are stored in uncompressed json format to `SAVE_DIR/NAME/MODEL/DATA/TRAINING_SETTING/seed-SEED/OUT_NAME-BATCH_ID.json`, where `BATCH_ID` denotes the statistics computed for batch number `BATCH_ID`.

## Monte Carlo Integration

For each measure, the number of Monte Carlo samples to be drawn for each training point is specified with the argument `--num-directions`. The number of training points to use is instead controlled by `--num-samples`. **Note:** `--batch-size` should divide `--num-samples`.

Furthermore, several strategies are available for generating Monte Carlo samples, and can be specified with the `--gen-strategy` command line argument:

1. `none`: disables Monte Carlo sampling (thus computing the metric only on training data).
2. `1px-shifts`: generate MC samples by randomly shifting each training sample by 1 pixel.
3. `4px-shifts`: generate MC samples by randomly shifting each training sample by 4 pixels.
4. `svd-5`: generate MC samples by randomly erasing 5 of the smallest singular values for each RBG channels of each training image (see our paper for details).
5. `svd-10`: generate MC samples by randomly erasing 10 of the smallest singular values for each RBG channels of each training image (see our paper for details).
6. `svd-15`: generate MC samples by randomly erasing 15 of the smallest singular values for each RBG channels of each training image (see our paper for details).
7. `svd-20`: generate MC samples by randomly erasing 20 of the smallest singular values for each RBG channels of each training image (see our paper for details).
8. `svd-22`: generate MC samples by randomly erasing 22 of the smallest singular values for each RBG channels of each training image (see our paper for details).
9. `svd-25`: generate MC samples by randomly erasing 25 of the smallest singular values for each RBG channels of each training image (see our paper for details).
10. `shifts-path`: for each training sample, generate `--num-directions` geodesic paths using successive 1-px shifts of the training sample. Each path is thus formed by `training sample -> 1px shift -> 2px shift -> 3px shift -> 4px shift`.
11. `svd-path`: for each training sample, generate `--num-directions` geodesic paths by connecting weak SVD augmentations. Each path is formed by `training samples -> erase 5 Singular Values -> erase 10 Singular Values -> ... -> erase 25 Singular Values`, where each weak augmentation is applied to the base training sample.

## Effective batch size and input data shape

Image data in our code is represented by tensors of shape `(*, C, H, W)`, where `C`, `H`, and `W` respectively denote the image' number of channels, height and width.

The batch size is controlled by one the choice of Monte Carlo integration strategy, as follows:

1. The argument `--batch-size B` controls how many training samples to use in a single forward-backward pass through the network.
2. The number of Monte Carlo samples `--num-directions M` to generate for each training sample.
3. If geodesic Monte Carlo integration is used, the effective batch size is multiplied by `A`, which is the length of each geodesic path, corresponding to how many weak augmentations are used to anchor the path to the data manifold local to each training sample. For instance geodesic paths generated using the strategy `--gen-strategy shifts-path` corresponds to `A = 5` weak augmentations per path (including the base training sample), while `--gen-strategy svd-path` corresponds to `A = 7`.

In summary, for any choice of batch size `--batch-size B`:
1. If `--gen-strategy none` is used, the resulting batch size is `B`, with input shape `(B, C, H, W)`.
2. If `--gen-strategy` is any of `1px-shifts, 4px-shifts, svd-5, svd-10, svd-15, svd-20, svd-22, svd-25` and `--num-directions M`, the resulting batch size `M * B`, with input shape `(M, B, C, H, W)`.
3. If `--num-directions M` and `--gen-strategy shifts-path`, the resulting batch size is `5 * M * B`, with input shape `(5, M, B, C, H, W)`.
4. If `--num-directions M` and `--gen-strategy svd-path`, the resulting batch size is `7 * M * B`, with input shape `(7, M, B, C, H, W)`.

The batch size `B` should be tuned according to the maximum batch size supported by the hardware in use.

## Normalization

Several strategies are available for normalizing the computed statistics, and can be specified with the argument `--normalization`:
- `None`: network output and Jacobian matrix are unnormalized (default).
- `softmax`: applies softmax to the network output, before computing the Jacobian w.r.t. the network's input.
- `logsoftmax`: applies logsoftmax to the network output, before computing the Jacobian w.r.t. the network's input.
- `crossentropy`: applies crossentropy to the network output, before computing the Jacobian w.r.t. the network's input.

In order to study the loss landscape, `--normalization crossentropy` should be used.

## Seeding

The following seeds should be set to control randomness:
- `--data-split-seed`: used for splitting train/validation set.
- `--label-seed`: used to corrupt training labels.
- `--mc-sample-seed`: used for Monte Carlo sampling of data augmentations (`compute_stats.py` only).
- `--seed`: used for initializing models during training, and creating and shuffling batches of data (`train.py` only).


# CONTRIBUTING

The source code is organized as follows:
- Model definitions can be found under `models`.
- Data augmentation transforms as well as label corruption algorithms are found under `core/data.py`.
- Metrics are defined `core/metrics.py`, with torch scripted helper functions available in `core/metric_helpers.py`.
- Data generation strategies are initialized in `core/strategies.py`.
- The main script for launching experiments is `compute_stats.py`.
