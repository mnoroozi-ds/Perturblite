"""Validated packet data and reproducible PerturbLite experiment splits.

Every sample contains 1,481 normalized byte features and a binary label.
Surrogate-classifier and generator data are kept disjoint. Each experiment
partition uses an 80/10/10 train/validation/test split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset


EXPECTED_FEATURES = 1481
DataPartition = Literal["classifier", "generator", "all"]


class PacketDataset(Dataset):
    """Packet features and labels backed by tensors."""

    def __init__(self, features: pd.DataFrame, labels: pd.Series):
        self.feature_names = list(features.columns)
        self.X = torch.as_tensor(features.to_numpy(dtype=np.float32), dtype=torch.float32)
        self.y = torch.as_tensor(labels.to_numpy(dtype=np.int64), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        return self.X[index], self.y[index]


@dataclass(frozen=True)
class PacketDataLoaders:
    train: DataLoader
    validation: DataLoader
    test: DataLoader
    feature_names: tuple[str, ...]


def _stratified_split(
    frame: pd.DataFrame,
    stratify: Sequence,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    left, right = train_test_split(
        frame,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    return left.reset_index(drop=True), right.reset_index(drop=True)


def _validate_frame(
    frame: pd.DataFrame,
    label_col: str,
    metadata_cols: Sequence[str],
) -> list[str]:
    if label_col not in frame.columns:
        raise ValueError(f"Label column '{label_col}' was not found")

    missing_metadata = set(metadata_cols) - set(frame.columns)
    if missing_metadata:
        raise ValueError(f"Metadata columns not found: {sorted(missing_metadata)}")

    feature_names = [
        column
        for column in frame.columns
        if column != label_col and column not in metadata_cols
    ]
    if len(feature_names) != EXPECTED_FEATURES:
        raise ValueError(
            f"PerturbLite requires exactly {EXPECTED_FEATURES} packet features; "
            f"found {len(feature_names)}"
        )

    labels = set(frame[label_col].dropna().unique().tolist())
    if not labels.issubset({0, 1}) or not labels:
        raise ValueError(f"'{label_col}' must contain binary labels 0 and 1; found {labels}")

    numeric = frame[feature_names].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        bad = numeric.columns[numeric.isna().any()].tolist()[:5]
        raise ValueError(f"Packet features must be numeric and non-null; invalid columns: {bad}")

    minimum = float(numeric.min().min())
    maximum = float(numeric.max().max())
    if minimum < 0.0 or maximum > 1.0:
        raise ValueError(
            "Packet byte features must already be normalized to [0, 1]; "
            f"observed range [{minimum:.6f}, {maximum:.6f}]"
        )
    frame.loc[:, feature_names] = numeric
    return feature_names


def _select_experiment_partition(
    frame: pd.DataFrame,
    label_col: str,
    partition: DataPartition,
    partition_col: str | None,
    classifier_fraction: float,
    random_state: int,
) -> pd.DataFrame:
    if partition == "all":
        return frame.reset_index(drop=True)

    if partition_col:
        if partition_col not in frame.columns:
            raise ValueError(f"Partition column '{partition_col}' was not found")
        selected = frame[frame[partition_col].astype(str).str.lower() == partition]
        if selected.empty:
            raise ValueError(
                f"No rows with {partition_col}='{partition}' were found"
            )
        return selected.reset_index(drop=True)

    if not 0.0 < classifier_fraction < 1.0:
        raise ValueError("classifier_fraction must be strictly between 0 and 1")

    classifier, generator = _stratified_split(
        frame,
        stratify=frame[label_col],
        test_size=1.0 - classifier_fraction,
        random_state=random_state,
    )
    return classifier if partition == "classifier" else generator


def build_packet_dataloaders(
    csv_path: str,
    label_col: str = "label",
    batch_size: int = 32,
    random_state: int = 42,
    num_workers: int = 0,
    partition: DataPartition = "all",
    partition_col: str | None = None,
    classifier_fraction: float = 0.5,
    folds: int = 5,
    fold_index: int = 0,
    malicious_only: bool = False,
    attack_type_col: str | None = None,
    attack_type: str | None = None,
) -> PacketDataLoaders:
    """Build validated 80/10/10 packet loaders from a CSV file.

    If ``partition_col`` is absent, a deterministic stratified split creates
    disjoint classifier and generator partitions.  A prepared data set may
    instead provide an explicit partition column containing ``classifier`` or
    ``generator``.
    """
    frame = pd.read_csv(csv_path)
    metadata_cols = [column for column in (partition_col, attack_type_col) if column]
    feature_names = _validate_frame(frame, label_col, metadata_cols)

    if attack_type is not None:
        if not attack_type_col:
            raise ValueError("attack_type requires attack_type_col")
        # An attack-specific binary data set contains all selected malicious
        # packets plus benign comparison packets. Benign rows commonly carry
        # an attack_type value such as "Benign", so equality-only filtering
        # would incorrectly discard the negative class.
        selected_attack = frame[attack_type_col].astype(str) == attack_type
        frame = frame[(frame[label_col] == 0) | selected_attack].reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"No samples found for attack type '{attack_type}'")
        if not (frame[label_col] == 1).any():
            raise ValueError(f"No malicious samples found for attack type '{attack_type}'")

    frame = _select_experiment_partition(
        frame,
        label_col=label_col,
        partition=partition,
        partition_col=partition_col,
        classifier_fraction=classifier_fraction,
        random_state=random_state,
    )

    if folds < 2:
        raise ValueError("folds must be at least 2")
    if not 0 <= fold_index < folds:
        raise ValueError(f"fold_index must be in [0, {folds - 1}]")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state + 1)
    selected_train = selected_holdout = None
    for index, (train_indices, holdout_indices) in enumerate(
        splitter.split(frame, frame[label_col])
    ):
        if index == fold_index:
            selected_train = train_indices
            selected_holdout = holdout_indices
            break
    train = frame.iloc[selected_train].reset_index(drop=True)
    holdout = frame.iloc[selected_holdout].reset_index(drop=True)
    validation, test = _stratified_split(
        holdout,
        stratify=holdout[label_col],
        test_size=0.5,
        random_state=random_state + 2,
    )

    if malicious_only:
        train = train[train[label_col] == 1].reset_index(drop=True)
        validation = validation[validation[label_col] == 1].reset_index(drop=True)
        test = test[test[label_col] == 1].reset_index(drop=True)
        if min(len(train), len(validation), len(test)) == 0:
            raise ValueError("Each generator split must contain malicious samples")

    def make_loader(split: pd.DataFrame, shuffle: bool) -> DataLoader:
        dataset = PacketDataset(split[feature_names], split[label_col])
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

    print(
        f"Packets ({partition}) - train: {len(train)} | validation: {len(validation)} "
        f"| test: {len(test)} | features: {len(feature_names)}"
    )
    return PacketDataLoaders(
        train=make_loader(train, shuffle=True),
        validation=make_loader(validation, shuffle=False),
        test=make_loader(test, shuffle=False),
        feature_names=tuple(feature_names),
    )

