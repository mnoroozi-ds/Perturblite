"""Dataset utilities for loading network-flow images.

Expected on-disk layout
-----------------------
A root directory with two sub-folders::

    data/
      0/   <- benign flow images  (class label 0)
      1/   <- malicious flow images (class label 1)

Each image file is a PNG/BMP representation of one network flow:
  - shape after ``transforms.ToTensor()`` : ``(3, 15, 1501)``
  - channel 0 = forward direction packets
  - channel 1 = backward direction packets
  - channel 2 = combined / metadata channel

Two dataset classes are provided
---------------------------------
``FlowImageDataset``
    One sample = one complete flow image ``(3, 15, 1501)``.  Used by the
    Generator training loop (AdvGAN) which needs the full 2-D flow.

``PacketDataset``
    One sample = one **packet** (a single non-zero row from channel 0),
    represented as a raw ``(1501,)`` byte vector.  Used for training and
    evaluating the surrogate classifier, since the paper is packet-based.
"""

import os
from typing import Optional

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


_DEFAULT_TRANSFORM = transforms.Compose([transforms.ToTensor()])


class FlowImageDataset(Dataset):
    """Dataset that wraps a list of ``(tensor, label)`` pairs.

    This is used after loading images from disk so that both the training
    and test sets can share the same ``DataLoader`` interface.

    Parameters
    ----------
    samples : list of [Tensor, Tensor]
        Each element is ``[image_tensor, label_tensor]`` as produced by
        :func:`load_samples_from_folder`.
    """

    def __init__(self, samples: list):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.samples[index]


class PacketDataset(Dataset):
    """Packet-level dataset: each non-zero packet row is an independent sample.

    Iterates over a list of flow image samples (as produced by
    :func:`load_samples_from_folder`) and extracts every non-zero packet row
    from the forward channel (channel 0).  Zero-padded rows (absent packets)
    are skipped.  The label from the parent flow is inherited by each packet.

    Parameters
    ----------
    flow_samples : list of [Tensor (3, 15, 1501), Tensor (scalar)]
        Flow-level samples as produced by :func:`load_samples_from_folder`.
    """

    def __init__(self, flow_samples: list):
        self.packets: list = []
        for img_tensor, label in flow_samples:
            # img_tensor: (3, 15, 1501) — use channel 0 (forward direction)
            fwd = img_tensor[0]          # (15, 1501)
            for row in range(fwd.shape[0]):
                pkt = fwd[row]           # (1501,)
                if pkt.abs().sum() > 0:  # skip zero-padded (absent) rows
                    self.packets.append((pkt, label))

    def __len__(self) -> int:
        return len(self.packets)

    def __getitem__(self, index: int):
        return self.packets[index]


def load_samples_from_folder(
    folder: str,
    label: int,
    transform=None,
) -> list:
    """Load all image files in *folder* and assign *label* to each.

    Parameters
    ----------
    folder : str
        Directory containing image files (PNG / BMP / JPG).
    label : int
        Integer class label (0 = benign, 1 = malicious).
    transform : callable, optional
        Torchvision transform to apply; defaults to ``transforms.ToTensor()``.

    Returns
    -------
    list of [Tensor, Tensor]
        Each element is ``[image_tensor, torch.tensor(label)]``.
    """
    if transform is None:
        transform = _DEFAULT_TRANSFORM

    samples = []
    for filename in sorted(os.listdir(folder)):
        file_path = os.path.join(folder, filename)
        try:
            img = Image.open(file_path).convert("RGB")
            samples.append([transform(img), torch.tensor(label)])
        except (OSError, Exception):
            # Skip unreadable files silently
            continue
    return samples


def build_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    test_size: float = 0.1,
    random_state: int = 42,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Build train and test DataLoaders from a class-folder data directory.

    The directory must contain ``0/`` and ``1/`` sub-folders holding the
    benign and malicious flow images respectively.  The combined set is
    split into train / test using *test_size*.

    Parameters
    ----------
    data_dir : str
        Root directory with sub-folders ``0`` and ``1``.
    batch_size : int
        Mini-batch size for both loaders.
    test_size : float
        Fraction of data reserved for the test set (e.g. 0.1 = 10%).
    random_state : int
        Seed for reproducible train/test splitting.
    num_workers : int
        Number of worker processes for DataLoader.

    Returns
    -------
    train_loader, test_loader : DataLoader, DataLoader
    """
    all_samples: list = []
    for label in (0, 1):
        folder = os.path.join(data_dir, str(label))
        if not os.path.isdir(folder):
            raise FileNotFoundError(
                f"Expected class folder not found: {folder}"
            )
        all_samples.extend(load_samples_from_folder(folder, label))

    train_list, test_list = train_test_split(
        all_samples, test_size=test_size, random_state=random_state
    )
    print(f"Train samples: {len(train_list)} | Test samples: {len(test_list)}")

    train_loader = DataLoader(
        FlowImageDataset(train_list),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        FlowImageDataset(test_list),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, test_loader


def build_packet_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    test_size: float = 0.1,
    random_state: int = 42,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Build packet-level train and test DataLoaders.

    Identical to :func:`build_dataloaders` but uses :class:`PacketDataset`
    so each batch contains individual packets ``(batch, 1501)`` rather than
    full flow images.  Use this for training and evaluating the surrogate.

    The train/test split is performed at the **flow level** before exploding
    into packets, so packets from the same flow never appear in both splits.

    Parameters
    ----------
    data_dir : str
        Root directory with sub-folders ``0`` and ``1``.
    batch_size : int
        Mini-batch size for both loaders.
    test_size : float
        Fraction of *flows* reserved for the test set.
    random_state : int
        Seed for reproducible splitting.
    num_workers : int
        Number of DataLoader worker processes.

    Returns
    -------
    train_loader, test_loader : DataLoader, DataLoader
    """
    all_samples: list = []
    for label in (0, 1):
        folder = os.path.join(data_dir, str(label))
        if not os.path.isdir(folder):
            raise FileNotFoundError(
                f"Expected class folder not found: {folder}"
            )
        all_samples.extend(load_samples_from_folder(folder, label))

    train_list, test_list = train_test_split(
        all_samples, test_size=test_size, random_state=random_state
    )

    train_pkt = PacketDataset(train_list)
    test_pkt  = PacketDataset(test_list)
    print(
        f"Flows  — train: {len(train_list)} | test: {len(test_list)}\n"
        f"Packets — train: {len(train_pkt)} | test: {len(test_pkt)}"
    )

    train_loader = DataLoader(
        train_pkt, batch_size=batch_size, shuffle=True,  num_workers=num_workers
    )
    test_loader = DataLoader(
        test_pkt,  batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader
