# Byzantine-Robust Federated Representation Learning

Code for **Byzantine-Robust Federated Representation Learning**.
Our approach learns a shared nonlinear representation through robust aggregation
while keeping a personalized linear head for every client. The baseline
robustly aggregates updates to one common model.

- [Paper](paper/byzantine_robust_federated_representation_learning.pdf)


## Installation

```bash
git clone https://github.com/LeoToso/adversarial-FL-nonlinear-rep.git
cd adversarial-FL-nonlinear-rep
conda create -n byzantine_rep python=3.10 -y
conda activate byzantine_rep
pip install -r requirements.txt
```

All commands below run from the repository root. Use `--device cuda` when a
CUDA GPU is available. Existing JSON/checkpoint pairs are skipped unless
`--overwrite` is supplied.

## Data

### CIFAR-10

`scripts/run_cifar10.py` downloads CIFAR-10 automatically and constructs 100
clients with two classes, 500 training images, and 100 test images per client.

### FEMNIST

Download the preprocessed LEAF FEMNIST JSON files into `data/femnist/{train,test}`,
then construct the 150-client, ten-letter partition:

```bash
python scripts/prepare_femnist.py \
  --leaf_dir data/femnist \
  --output_dir data/femnist_partition \
  --published_compatibility
```

### School Exam

The School Exam file is downloaded automatically from the MALSAR repository.
Each of the 139 schools is one client; each run samples 20 honest schools per
round and appends five Byzantine updates.

## Reproduce Table 1: classification

Cross-entropy (baseline and representation learning):

```bash
python scripts/run_cifar10.py --mode robust \
  --loss_types cross_entropy \
  --honest_per_round 50 \
  --algorithms baseline representation_learning \
  --aggregators NNM+Krum NNM+TrMean \
  --attacks ALIE Mimic --seeds 42 123 456 --device cuda

python scripts/run_femnist.py --mode robust \
  --partition data/femnist_partition \
  --loss_type cross_entropy \
  --honest_per_round 50 \
  --algorithms baseline representation_learning \
  --aggregators NNM+Krum NNM+TrMean \
  --attacks ALIE Mimic --seeds 42 123 456 --device cuda
```

Multiclass squared loss (representation learning):

```bash
python scripts/run_cifar10.py --mode robust \
  --loss_types multiclass_ls --honest_per_round 50 \
  --algorithms representation_learning \
  --aggregators NNM+Krum NNM+TrMean \
  --attacks ALIE Mimic --seeds 42 123 456 --device cuda

python scripts/run_femnist.py --mode robust \
  --partition data/femnist_partition --loss_type multiclass_ls \
  --honest_per_round 50 --algorithms representation_learning \
  --aggregators NNM+Krum NNM+TrMean \
  --attacks ALIE Mimic --seeds 42 123 456 --device cuda
```

| Dataset | Loss | Method | Krum/ALIE | Krum/Mimic | TrMean/ALIE | TrMean/Mimic |
|---|---|---|---:|---:|---:|---:|
| CIFAR-10 | Cross-entropy | Baseline | 50.46 ± 0.57 | 46.39 ± 1.76 | 50.46 ± 0.71 | 47.17 ± 1.58 |
| CIFAR-10 | Cross-entropy | Rep. learning | **88.79 ± 0.42** | **88.50 ± 0.59** | **88.76 ± 0.44** | **88.54 ± 0.61** |
| CIFAR-10 | Multiclass | Rep. learning | 88.33 ± 0.53 | 87.95 ± 0.51 | 88.42 ± 0.52 | 87.89 ± 0.60 |
| FEMNIST | Cross-entropy | Baseline | 72.95 ± 1.95 | 70.34 ± 1.23 | 73.09 ± 2.03 | 70.66 ± 1.47 |
| FEMNIST | Cross-entropy | Rep. learning | 91.89 ± 0.31 | 92.43 ± 0.15 | 92.02 ± 0.28 | 92.51 ± 0.07 |
| FEMNIST | Multiclass | Rep. learning | **93.56 ± 0.24** | **93.60 ± 0.07** | **93.57 ± 0.22** | **93.67 ± 0.13** |

Entries are mean local test accuracy (%) ± sample standard deviation over seeds
42, 123, and 456. Classification results average the final ten rounds.

## Reproduce Figure 1: FEMNIST client scaling

Run the adversarial curves and the no-attack reference:

```bash
python scripts/run_femnist.py --mode robust \
  --partition data/femnist_partition --loss_type cross_entropy \
  --honest_per_round 10 20 50 \
  --algorithms baseline representation_learning \
  --aggregators NNM+Krum NNM+TrMean --attacks Mimic \
  --seeds 42 123 456 --device cuda

python scripts/run_femnist.py --mode clean \
  --partition data/femnist_partition --loss_type cross_entropy \
  --honest_per_round 50 \
  --algorithms baseline representation_learning \
  --seeds 42 123 456 --device cuda
```

The publication-ready figure is included at
`figures/femnist_cross_entropy.pdf`.

## Reproduce Table 2: School Exam

```bash
python scripts/run_school.py --device cuda \
  --seeds 42 123 456 \
  --algorithms baseline representation_learning \
  --aggregators NNM+Krum NNM+TrMean \
  --attacks ALIE Mimic
```

| Method | Krum/ALIE | Krum/Mimic | TrMean/ALIE | TrMean/Mimic |
|---|---:|---:|---:|---:|
| Baseline | 0.7611 ± 0.0452 | 0.6532 ± 0.0201 | 0.7672 ± 0.0465 | 0.6531 ± 0.0226 |
| Rep. learning | **0.6324 ± 0.0272** | **0.6300 ± 0.0276** | **0.6316 ± 0.0275** | **0.6301 ± 0.0276** |

Constant-predictor sanity checks are `0.9913 ± 0.0306` for the pooled training
mean and `0.8826 ± 0.0263` for the per-school training mean. Lower MSE is better.

## Tests

```bash
python -m py_compile core/*.py experiments/*.py scripts/*.py
python -m pytest -q
```

## Citation

The arXiv identifier will be added after publication. Until then, please cite
the bundled manuscript and this repository.
