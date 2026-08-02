import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from utils.dataset import EXPECTED_FEATURES, build_packet_dataloaders


class DatasetTests(unittest.TestCase):
    def test_disjoint_partition_and_80_10_10_split(self):
        rows = 100
        columns = ["ip_header_byte_9", "tcp_header_byte_14"] + [
            f"tcp_segment_data_byte_{index}" for index in range(EXPECTED_FEATURES - 2)
        ]
        values = np.zeros((rows, EXPECTED_FEATURES), dtype=np.float32)
        frame = pd.DataFrame(values, columns=columns)
        frame["label"] = np.tile([0, 1], rows // 2)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "packets.csv"
            frame.to_csv(path, index=False)
            loaders = build_packet_dataloaders(
                str(path), batch_size=8, partition="classifier", random_state=7
            )

        # Default classifier partition is half of 100 rows, then 80/10/10.
        self.assertEqual(len(loaders.train.dataset), 40)
        self.assertEqual(len(loaders.validation.dataset), 5)
        self.assertEqual(len(loaders.test.dataset), 5)
        self.assertEqual(len(loaders.feature_names), EXPECTED_FEATURES)


if __name__ == "__main__":
    unittest.main()
