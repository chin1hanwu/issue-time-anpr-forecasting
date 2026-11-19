"""Check that case-weighted and period-weighted service KPIs are distinct.

Adapted from the private-repo test of the same name: the private helper
``stratified_kpi_with_fallback`` is not part of this public release, so the
test targets the public equivalent, ``service_rule_metrics`` in
``frozen_evaluation.py``, with the same synthetic setup.
"""
import unittest

import numpy as np
import pandas as pd

from frozen_evaluation import service_rule_metrics


class ServiceKpiGranularityTest(unittest.TestCase):
    def test_case_and_period_weighting_are_distinct(self):
        period_ids = np.array([0, 0, 0, 1, 2])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        component_lo = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
        component_hi = np.array([2.0, 3.0, 2.0, 3.0, 10.0])
        fallback_lo = np.zeros(5)
        fallback_hi = np.full(5, 10.0)

        test = {
            "y_raw": y,
            "metadata": pd.DataFrame({"update_period_id": period_ids}),
        }
        # Period-level publication decisions: periods 0 and 1 broadcast,
        # period 2 falls back. (Four of the five cases are broadcast.)
        modes = pd.Series([True, True, False], index=[0, 1, 2])
        result, _, _, _ = service_rule_metrics(
            test, component_lo, component_hi, fallback_lo, fallback_hi, modes
        )

        self.assertAlmostEqual(result["publication_rate_period"], 2.0 / 3.0)
        self.assertAlmostEqual(result["trajectory_exposure_share"], 4.0 / 5.0)
        # Combined coverage: case-weighted 3/5, period-weighted 5/9.
        self.assertAlmostEqual(result["combined_coverage_case"], 3.0 / 5.0)
        self.assertAlmostEqual(result["combined_coverage_period"], 5.0 / 9.0)
        self.assertNotEqual(
            result["combined_coverage_case"], result["combined_coverage_period"]
        )
        # The same distinction holds for the broadcast-only intervals.
        self.assertAlmostEqual(result["broadcast_coverage_case"], 2.0 / 4.0)
        self.assertAlmostEqual(result["broadcast_coverage_period"], 1.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
