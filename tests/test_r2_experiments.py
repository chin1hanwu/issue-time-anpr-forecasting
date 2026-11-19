"""Check the boundary conventions of the second-revision comparisons that are easy to change by accident."""
import unittest

import numpy as np
import pandas as pd

from r2_experiment_runner import block_indices, gate_mask, interval_values, pm, scale_for_coverage


class R2ExperimentTests(unittest.TestCase):
    def test_blocks_respect_cadence_and_gaps(self):
        for cadence in [2, 5, 10]:
            times = list(pd.date_range("2020-01-01", periods=5, freq=f"{cadence}min"))
            times += list(pd.date_range("2020-01-02", periods=6, freq=f"{cadence}min"))
            meta = pd.DataFrame({"update_period_id": np.arange(11), "prediction_time": times})
            periods, indices = block_indices(meta, cadence, 3, repeats=25)
            self.assertEqual(indices.shape, (25, 11))
            for row in indices:
                for offset in range(0, 11, 3):
                    selected = periods.iloc[row[offset:offset + 3]].prediction_time
                    self.assertTrue((selected.diff().dropna() == pd.Timedelta(minutes=cadence)).all())

    def test_scale_uses_equal_period_weight(self):
        y = np.array([0.0] * 9 + [10.0])
        lo, hi = np.full(10, -1.0), np.ones(10)
        meta = pd.DataFrame({"case_id": range(10), "update_period_id": [0] * 9 + [1]})
        scale = scale_for_coverage(y, lo, hi, meta, 0.75)
        self.assertGreaterEqual(scale, 10.0)
        covered = ((y >= -scale) & (y <= scale)).astype(float)
        self.assertGreaterEqual(pm(covered, meta).mean(), 0.75)
        self.assertEqual(pm(((y >= -1) & (y <= 1)).astype(float), meta).mean(), 0.5)

    def test_score_and_tail_misses(self):
        values = interval_values(np.array([-1.0, 1.0, 3.0]), np.zeros(3), np.full(3, 2.0))
        np.testing.assert_array_equal(values[:, 2], [22.0, 2.0, 22.0])
        np.testing.assert_array_equal(values[:, 3], [1.0, 0.0, 0.0])
        np.testing.assert_array_equal(values[:, 4], [0.0, 0.0, 1.0])

    def test_threshold_equality_is_broadcast(self):
        state = pd.DataFrame({"R_comp": [0.5, 0.49, 0.8], "rho_t": [0.3, 0.2, 0.31]})
        mask = gate_mask(state, {"mode": "joint_gate", "R_low": 0.5, "rho_high": 0.3})
        np.testing.assert_array_equal(mask, [True, False, False])
        self.assertFalse(gate_mask(state, {"mode": "fallback_only"}).any())

    def test_empty_block_support_fails(self):
        meta = pd.DataFrame({"update_period_id": [0, 1], "prediction_time": pd.to_datetime(["2020-01-01", "2020-01-02"])})
        with self.assertRaises(ValueError):
            block_indices(meta, 5, 2)


if __name__ == "__main__":
    unittest.main()
