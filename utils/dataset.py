"""Dataset utilities for loading network-flow images.

Expected on-disk layout
-----------------------
Each split (train / test) is a directory with two sub-folders::

    data/
      train/
        0/   <- benign flows (class 0)
        1/   <- malicious flows (class 1)
      test/
        0/
        1/

Each image file is a PNG/BMP representation of one network flow:
  - shape after ``transforms.ToTensor()`` : ``(3, 15, 1501)``
  - channel 0 = forward direction packets
  - channel 1 = backward direction packets
  - channel 2 = combined / metadata channel

Images in sub-folder ``0`` receive label ``0`` (benign);
images in sub-folder ``1`` receive label ``1`` (malicious).
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
