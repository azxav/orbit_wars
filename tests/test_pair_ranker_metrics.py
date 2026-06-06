from __future__ import annotations

import unittest

import numpy as np

from train_pair_ranker import binary_auc, top1_target_accuracy


class PairRankerMetricsTest(unittest.TestCase):
    def test_binary_auc_handles_ordered_scores(self) -> None:
        y_true = np.asarray([0, 0, 1, 1], dtype=np.float32)
        y_score = np.asarray([0.1, 0.2, 0.8, 0.9], dtype=np.float32)

        self.assertEqual(binary_auc(y_true, y_score), 1.0)

    def test_top1_target_accuracy_scores_highest_candidate_per_group(self) -> None:
        rows = [
            {"group_uid": "g1", "label": 0},
            {"group_uid": "g1", "label": 1},
            {"group_uid": "g2", "label": 1},
            {"group_uid": "g2", "label": 0},
        ]
        scores = np.asarray([0.2, 0.9, 0.4, 0.8], dtype=np.float32)

        self.assertEqual(top1_target_accuracy(rows, scores), 0.5)


if __name__ == "__main__":
    unittest.main()
