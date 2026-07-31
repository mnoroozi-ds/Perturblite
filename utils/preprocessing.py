"""Preprocessing note.

Features are **pre-extracted** by the data generation pipeline: each CSV
row already contains the 1481 feature values (immutable header bytes
stripped).  No column stripping is performed at runtime.

This module provides only the **mutable-position mask** for the Generator:
which of the 1481 feature positions correspond to fields that can safely
be perturbed without breaking packet validity.

Mutable positions in the 1481-feature vector
---------------------------------------------
The original byte indices (0-based within a 1501-byte packet) that are
mutable, and their mapped indices after stripping:

  Original byte 9  (IP TTL)              → feature index  7
  Original byte 35 (TCP urgent ptr, high) → feature index 17
  Original byte 36 (TCP urgent ptr, low)  → feature index 18

Mapping formula (stripping cols 1-2, 10-25, 37-38)::

  0     → 0
  3-9   → 1-7    (col 9 / TTL → index 7)
  26-36 → 8-18   (col 35 → 17, col 36 → 18)
  39-1500 → 19-1480

Note: if your feature extraction pipeline uses a different column order,
update ``MUTABLE_FEATURE_POSITIONS`` in ``attack/masks.py`` accordingly.
"""
# No runtime functions needed — features arrive ready-to-use from CSV.
