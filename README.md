# PerturbLite

PyTorch implementation of **PerturbLite**, a constrained, autoencoder-based
red-teaming method for packet-level network intrusion detection systems.

## What the implementation models

Each example is one attacker-controlled, forward TCP packet represented by
exactly **1,481 normalized byte features** and a binary label (`0` benign,
`1` malicious). The external PCAP preparation pipeline:

1. extracts and labels forward packets;
2. zero-pads packets to a common size of at most 1,594 bytes;
3. removes the 14-byte Ethernet header;
4. removes IP version, differentiated services, protocol, source and
   destination IP addresses;
5. removes TCP source and destination ports;
6. removes IP/TCP options and checksums; and
7. converts the remaining 1,481 bytes to decimal and normalizes them to
   `[0, 1]`.

That PCAP-to-CSV extraction step is not included here. Checksums therefore do
not appear in the model input. Payload bytes remain classifier features but
are immutable during adversarial generation.

## CSV schema

The constraint code derives packet semantics from feature names:

```text
ip_header_byte_...          retained IP header feature (mutable)
tcp_header_byte_...         retained TCP header feature (mutable)
tcp_segment_data_byte_...   TCP payload/segment feature (immutable)
label                       0 or 1
```

There must be exactly 1,481 feature columns, all numeric and normalized to
`[0, 1]`. For another naming scheme, pass the exact mutable header column names
to `train_attack.py --mutable-features name1,name2,...`. Payload names are
always protected from mutation.

An optional partition column can explicitly mark rows as `classifier` or
`generator`. This is the preferred way to reproduce a prepared experimental
split. If omitted, both scripts deterministically create disjoint stratified
partitions using `--classifier-fraction` (default `0.5`). Each partition is
subsequently split 80%/10%/10% for training, validation, and testing. The
default five folds reproduce the cross-validation protocol; select a fold with
`--fold-index 0` through `--fold-index 4`.

## Install

```bash
python -m pip install -r requirements.txt
```

Python 3.10 or newer and PyTorch 2.0 or newer are required.

## Train the surrogate

Train one binary surrogate per attack category (or use a prepared AllClasses
CSV):

```bash
python train_classifier.py \
  --data data/dos_packets.csv \
  --epochs 200 \
  --patience 20 \
  --save-path checkpoints/dos_classifier.pth
```

If several attack categories share one CSV, add:

```bash
--attack-type-col attack_type --attack-type DoS
```

The surrogate architecture is `1481 -> 2048 -> 1024 -> 512 -> 256 -> 64 -> 1`
with ReLU, dropout `0.1`, and a sigmoid output.

## Train an attack-specific generator

```bash
python train_attack.py \
  --data data/dos_packets.csv \
  --classifier-path checkpoints/dos_classifier.pth \
  --epochs 1000 \
  --alpha 0.95 \
  --beta 0.05 \
  --checkpoint-dir checkpoints/dos
```

Only malicious samples train the generator. Benign samples from its training
partition derive each mutable feature's clipping interval:

```text
[minimum benign value, 99th percentile benign value]
```

For generator output `G(x)`, the constrained packet follows Algorithm 1:

```text
x_adv = immutable_mask * x
      + mutable_mask * clip(G(x), lower_bounds, upper_bounds)
```

The loss combines normalized classifier-deception and squared-perturbation
terms using `alpha=0.95` and `beta=0.05` by default.

## Evaluate

```bash
python evaluate.py \
  --data data/dos_packets.csv \
  --classifier-path checkpoints/dos_classifier.pth \
  --generator-path checkpoints/dos/G_best.pth
```

ASR is measured only over genuine malicious packets correctly detected before
perturbation:

```text
ASR = clean true-positive attacks changed to benign / clean true-positive attacks
```

Evaluation also reports clean accuracy, adversarial accuracy, and MSE across
mutable features of malicious packets.

## Verification

```bash
python -m unittest discover -s tests -v
python -m compileall -q .
```

## Constraint and training guarantees

- The generator is a convolutional autoencoder with no discriminator.
- Header masks are derived from semantic feature names.
- Payload is included as classifier input but copied unchanged.
- Checksums and options are preprocessing exclusions, not runtime perturbations.
- Generated mutable header values are bounded by their benign feature ranges.
- Generator training uses malicious packets only and a normalized
  dual-objective loss.
