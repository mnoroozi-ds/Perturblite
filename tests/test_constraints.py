import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from attack.masks import (
    build_feature_masks,
    compute_benign_feature_bounds,
    constrain_generated_sample,
)


class ConstraintTests(unittest.TestCase):
    def setUp(self):
        self.names = (
            "ip_header_byte_9",
            "tcp_header_byte_14",
            "tcp_segment_data_byte_1",
            "other_immutable",
        )

    def test_semantic_mask_never_selects_payload(self):
        immutable, mutable = build_feature_masks(self.names)
        self.assertEqual(mutable.tolist(), [1.0, 1.0, 0.0, 0.0])
        self.assertTrue(torch.equal(immutable + mutable, torch.ones(4)))

    def test_algorithm_one_preserves_immutable_and_clips_mutable(self):
        immutable, mutable = build_feature_masks(self.names)
        original = torch.tensor([[0.4, 0.5, 0.6, 0.7]])
        generated = torch.tensor([[0.9, 0.1, 0.0, 0.0]])
        lower = torch.tensor([0.2, 0.3, 0.0, 0.0])
        upper = torch.tensor([0.8, 0.6, 1.0, 1.0])
        crafted = constrain_generated_sample(
            original, generated, immutable, mutable, lower, upper
        )
        self.assertTrue(torch.allclose(crafted, torch.tensor([[0.8, 0.3, 0.6, 0.7]])))

    def test_bounds_use_only_benign_samples(self):
        features = torch.tensor(
            [[0.2, 0.4, 0.1, 0.1], [0.4, 0.6, 0.2, 0.2], [1.0, 1.0, 0.3, 0.3]]
        )
        labels = torch.tensor([0, 0, 1])
        loader = DataLoader(TensorDataset(features, labels), batch_size=3)
        _, mutable = build_feature_masks(self.names)
        lower, upper = compute_benign_feature_bounds(loader, mutable, upper_quantile=1.0)
        self.assertTrue(torch.allclose(lower[:2], torch.tensor([0.2, 0.4])))
        self.assertTrue(torch.allclose(upper[:2], torch.tensor([0.4, 0.6])))


if __name__ == "__main__":
    unittest.main()
