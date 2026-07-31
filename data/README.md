# Dataset

This directory is the expected root for your network-flow image dataset.

## Required Layout

```
data/
  0/          ← benign flow images (class label 0)
    flow_0001.png
    flow_0002.png
    ...
  1/          ← malicious / attack flow images (class label 1)
    flow_0001.png
    flow_0002.png
    ...
```

## Image Specification

| Property | Value |
|----------|-------|
| Format   | PNG, BMP, or JPG |
| Channels | 3 (RGB) |
| Height   | 15 pixels (one row per packet) |
| Width    | 1501 pixels (one column per payload byte) |
| Bit depth | 8-bit per channel (pixel values 0–255 = raw byte values) |

## Encoding

Each pixel value directly encodes a **raw byte value** (0–255) from the network packet capture:
- **Row** i = packet i within the flow (up to 15 packets; zero-padded if fewer)
- **Column** j = byte j of the packet
- **Channel 0** = forward direction packets
- **Channel 1** = backward direction packets
- **Channel 2** = combined / metadata channel

## Generating the Dataset

Flow images must be generated from a packet capture (PCAP) file using a separate
preprocessing pipeline (not included in this repository). The recommended tool is
[CICFlowMeter](https://github.com/ahlashkari/CICFlowMeter) or a custom Scapy script
that segments per-flow packets into fixed-size image arrays.

## Notes

- Rows with no packet (padding) should be set to all zeros.
- Byte values should **not** be normalised before saving — `transforms.ToTensor()`
  in `utils/dataset.py` automatically scales to [0, 1] at load time.
- The `build_dataloaders()` function automatically performs a 90/10 train/test split.
