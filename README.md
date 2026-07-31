# Adversarial Traffic Attack

A PyTorch implementation of an **AdvGAN-based adversarial attack** against a CNN binary classifier trained on raw network-flow packet data.

The project demonstrates that small, targeted perturbations applied to specific mutable header fields (IP TTL, TCP urgent pointer) can reliably fool a deep learning intrusion-detection classifier while keeping modified packets structurally valid.

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
| `BinaryClassifier` | 4-layer CNN trained to classify network flows as benign (0) or malicious (1) |
| `Generator` | Encoder–ResNet–Decoder network that produces per-flow perturbations |
| `AdvGAN` | Adversarial training loop that minimises BCE loss of the *perturbed* flow's classification |
| Checksum utilities | RFC 791 / RFC 793 implementations to repair IP/TCP checksums after perturbation |

**Input representation**: each network flow is encoded as a 3-channel image of shape `(3, 15, 1501)`:
- **3 channels**: forward direction / backward direction / combined flow
- **15 rows**: up to 15 packets per flow
- **1501 columns**: byte values (0–255) per packet

---

## Repository Structure

```
adversarial-traffic-attack/
├── README.md
├── requirements.txt
├── data/
│   └── README.md          ← expected dataset layout
├── models/
│   ├── classifier.py      ← BinaryClassifier CNN
│   ├── generator.py       ← Generator (Encoder–ResNet–Decoder)
│   └── discriminator.py   ← Discriminator (unused in default training)
├── attack/
│   ├── adv_gan.py         ← AdvGAN training class
│   └── masks.py           ← mutable byte-position masks
├── utils/
│   ├── dataset.py         ← FlowImageDataset, DataLoader builders
│   ├── preprocessing.py   ← strip immutable header columns
│   └── checksum.py        ← IP / TCP checksum calculation
├── train_classifier.py    ← CLI: train binary classifier
├── train_attack.py        ← CLI: train AdvGAN generator
└── evaluate.py            ← CLI: measure attack success rate
```

---

## Dataset Format

See [data/README.md](data/README.md) for the full specification.

Each flow image is a PNG/BMP file with pixel values representing raw byte values (0–255) of network packets. Images are organised into class sub-folders:

```
data/
  0/   ← benign flows
  1/   ← malicious / attack flows
```

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
    --epochs 1000 \
    --batch-size 32 \
    --save-path best_classifier.pth
```

The script splits data 90/10 into train/test, reports per-epoch loss and accuracy, and saves the best checkpoint to `--save-path`.

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

Reports **clean accuracy**, **adversarial accuracy**, and **attack success rate (ASR)**.

---

## Architecture

### BinaryClassifier

```
Conv(3→96, 5×5) → ReLU → MaxPool → BN
Conv(96→128, 5×5) → ReLU → MaxPool
Conv(128→192, 3×3) → ReLU → MaxPool
Conv(192→384, 4×4) → ReLU → MaxPool → Dropout(0.2)
Flatten → FC(109440→64) → ReLU → Dropout → FC(64→1) → Sigmoid
```

### Generator

```
Encoder:    Conv(1→32→128, kernel=1, InstanceNorm, ReLU)
Bottleneck: 4× ResNet blocks (reflection padding, BN, ReLU)
Decoder:    ConvTranspose(128→64→32→1, InstanceNorm, Tanh)
```

---

## Design Choices

- **Mutable-only perturbation**: the mask restricts changes to columns 9 (TTL), 35, and 36 (TCP urgent pointer / options), leaving routing-critical fields intact.
- **Multiplicative application**: `perturbed = G(x) * mask * x + x` scales the perturbation relative to each byte's value.
- **Discriminator disabled**: the adversarial (GAN) discriminator loss is commented out in the default configuration; only the adversarial BCE loss from the frozen classifier is used.
- **Checksum repair**: `utils/checksum.py` provides RFC 791/793 checksum functions for post-hoc validity correction of perturbed packets.
