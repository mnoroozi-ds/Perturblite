# Adversarial Traffic Attack

A PyTorch implementation of an **PerturbLite** against a surrogate MLP classifier trained on raw network-flow packet data.

The project demonstrates that small, targeted perturbations applied to specific mutable header fields (IP TTL, TCP urgent pointer) can reliably fool a deep learning intrusion-detection model while keeping modified packets structurally valid.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Dataset Format](#dataset-format)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Train the Classifier](#1-train-the-classifier)
  - [2. Train the Attack Generator](#2-train-the-attack-generator)
  - [3. Evaluate the Attack](#3-evaluate-the-attack)
- [Architecture](#architecture)
- [Design Choices](#design-choices)

---

## Overview

| Component | Description |
|-----------|-------------|
| `BinaryClassifier` | 5-hidden-layer MLP surrogate (1481 → 2048 → … → 64 → 1) trained to classify network flows as benign (0) or malicious (1) |
| `Generator` | 1-D convolutional encoder–decoder that produces per-flow byte perturbations (61,953 parameters) |
| `AdvGAN` | Adversarial training loop that minimises BCE loss of the *perturbed* flow's classification via the surrogate |
| Checksum utilities | RFC 791 / RFC 793 implementations to repair IP/TCP checksums after perturbation |

***Input representation**: the data is **packet-based** — each sample is one packet
represented as a pre-extracted **1481-feature** vector (CSV row). The data generation
pipeline strips immutable header bytes (IP/TCP checksums, sequence numbers) before
saving. Labels (0 = benign, 1 = malicious) are included as a column in the same CSV.

---

## Repository Structure

```
adversarial-traffic-attack/
├── README.md
├── requirements.txt
├── data/
│   └── README.md          ← expected dataset layout
├── models/
│   ├── classifier.py      ← BinaryClassifier MLP surrogate
│   ├── generator.py       ← Generator (Conv1d encoder–decoder, 61 953 params)
│   └── discriminator.py   ← Discriminator (unused in default training)
├── attack/
│   ├── adv_gan.py         ← AdvGAN training class
│   └── masks.py           ← mutable byte-position masks
├── utils/
│   ├── dataset.py         ← FlowImageDataset, DataLoader builders
│   ├── preprocessing.py   ← flatten_for_surrogate / extract_generator_channel
│   └── checksum.py        ← IP / TCP checksum calculation
├── train_classifier.py    ← CLI: train binary classifier
├── train_attack.py        ← CLI: train AdvGAN generator
└── evaluate.py            ← CLI: measure attack success rate
```

---

## Dataset Format

See [data/README.md](data/README.md) for the full specification.

A single CSV file where each row is one packet:

```
feat_0, feat_1, ..., feat_1480, label
0.12,   0.00,   ..., 0.87,      0
0.45,   0.11,   ..., 0.23,      1
```

- **1481 columns**: pre-extracted feature values (immutable header bytes already stripped)
- **label column**: 0 = benign, 1 = malicious

---

## Installation

```bash
git clone https://github.com/<your-username>/adversarial-traffic-attack.git
cd adversarial-traffic-attack
pip install -r requirements.txt
```

Python ≥ 3.10 and PyTorch ≥ 2.0 are required.

---

## Usage

### 1. Train the Classifier

```bash
python train_classifier.py \
    --data-dir data/ \
    --epochs 200 \
    --batch-size 32 \
    --lr 1e-4 \
    --save-path best_classifier.pth
```

The script loads the CSV via `build_packet_dataloaders`, splits 90/10 into train/test (stratified), trains the surrogate on 1481-feature packet vectors, and saves the best checkpoint.

### 2. Train the Attack Generator

```bash
python train_attack.py \
    --data-dir data/ \
    --classifier-path best_classifier.pth \
    --epochs 200 \
    --batch-size 32 \
    --adv-lambda 10 \
    --checkpoint-every 20
```

Checkpoints are saved to `checkpoints/G_epoch_<N>.pth` every 20 epochs and a final `checkpoints/G_final.pth` at the end.

### 3. Evaluate the Attack

```bash
python evaluate.py \
    --data-dir data/ \
    --classifier-path best_classifier.pth \
    --generator-path checkpoints/G_final.pth
```

Reports **clean accuracy**, **adversarial accuracy**, and the **Attack Success Rate (ASR)**:

$$\text{ASR} = \frac{T_m}{TP}$$

where $TP$ is the number of samples correctly classified on clean input and $T_m$ is the subset of those that the adversarial perturbation causes to be misclassified.

---

## Architecture

### Surrogate Classifier (MLP)

```
Input  1481
FC(2048) + Dropout(0.1) + ReLU
FC(1024) + Dropout(0.1) + ReLU
FC(512)  + Dropout(0.1) + ReLU
FC(256)  + Dropout(0.1) + ReLU
FC(64)   + Dropout(0.1) + ReLU
FC(1)    + Sigmoid
```

Hyperparameters: Adam · lr = 1e-4 · epochs = 200

### Generator (1-D Conv encoder–decoder)

```
Encoder:  Conv1d(1→32, k=3) + BN + ReLU
          Conv1d(32→64, k=3) + BN + ReLU
          Conv1d(64→128, k=3) + BN + ReLU
Decoder:  ConvTr1d(128→64, k=3) + BN + ReLU
          ConvTr1d(64→32, k=3) + BN + ReLU
          ConvTr1d(32→1, k=3)  + BN + Sigmoid
```

Parameters: **61,953** · Hyperparameters: Adam · lr = 1e-4 · epochs = 1000

Each packet row is processed independently as a 1-D byte sequence; the forward pass reshapes `(batch, 1, 15, 1501)` → `(batch×15, 1, 1501)` through the Conv1d layers, then restores the original shape.

---

## Design Choices

- **Mutable-only perturbation**: the mask restricts changes to columns 9 (TTL), 35, and 36 (TCP urgent pointer / options), leaving routing-critical fields intact.
- **Multiplicative application**: `perturbed = G(x) * mask * x + x` scales the perturbation relative to each byte's value.
- **CSV packet data**: each row in the CSV is one packet. Features are pre-extracted by the data generation pipeline; no runtime byte-stripping is needed.
- **Flat surrogate input**: the surrogate receives 1481-feature vectors directly from the CSV. Mutable positions (feature indices 7, 17, 18 → IP TTL, TCP urgent pointer) are masked for perturbation.
- **Discriminator disabled**: the adversarial (GAN) discriminator loss is commented out in the default configuration; only the adversarial BCE loss from the frozen surrogate is used.
- **Checksum repair**: `utils/checksum.py` provides RFC 791/793 checksum functions for post-hoc validity correction of perturbed packets.
