"""Dataset utilities for loading per-packet CSV data.

Expected data format
--------------------
A single CSV file where:
  - Each row is one packet
  - There is one label column (default: ``'label'``) containing 0 (benign)
    or 1 (malicious)
  - All remaining columns are the 1481 pre-extracted feature values
    (header bytes already stripped by the data generation pipeline)

Example structure::

    feat_0, feat_1, ..., feat_1480, label
    0.12,   0.00,   ..., 0.87,      0
    0.45,   0.11,   ..., 0.23,      1
    ...

The train/test split is performed at load time using scikit-learn's
``train_test_split``.
"""

import os

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class PacketDataset(Dataset):
    """Packet-level dataset backed by a pandas DataFrame.

    Each item is ``(features, label)`` where features is a float32
    Tensor of shape ``(n_features,)`` and label is a scalar int64 Tensor.

    Parameters
    ----------
    features : pd.DataFrame or np.ndarray  shape (N, n_features)
    labels   : pd.Series or np.ndarray     shape (N,)
    """

    def __init__(self, features, labels):
        self.X = torch.tensor(features.values if hasattr(features, 'values') else features,
                              dtype=torch.float32)
        self.y = torch.tensor(labels.values   if hasattr(labels,   'values') else labels,
                              dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        return self.X[index], self.y[index]


def build_packet_dataloaders(
    csv_path: str,
    label_col: str = 'label',
    batch_size: int = 32,
    test_size: float = 0.1,
    random_state: int = 42,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Build train and test DataLoaders from a per-packet CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.  Must contain *label_col* plus 1481 feature
        columns.
    label_col : str
        Name of the label column (default: ``'label'``).
    batch_size : int
        Mini-batch size for both loaders.
    test_size : float
        Fraction of packets reserved for the test set.
    random_state : int
        Seed for reproducible splitting.
    num_workers : int
        Number of DataLoader worker processes.

    Returns
    -------
    train_loader, test_loader : DataLoader, DataLoader
    """
    df = pd.read_csv(csv_path)

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in {csv_path}")

    X = df.drop(columns=[label_col])
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(
        f"Packets — train: {len(X_train)} | test: {len(X_test)} | "
        f"features: {X_train.shape[1]}"
    )

    train_loader = DataLoader(
        PacketDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        PacketDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Alias kept for the AdvGAN training loop (uses flow images from disk)
# ---------------------------------------------------------------------------
build_dataloaders = build_packet_dataloaders

import os

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
