import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checks import model_score
from plots.evaluation import final_store_factors, validation_model_values


class ModelPlotConsistency(unittest.TestCase):
    def test_plot_and_runner_scores_match(self):
        for tile, depth, bandwidth_ratio, gamma in (
            ((96, 71, 70), 5, 0.20792, 1.0088),
            ((96, 82, 74), 9, 0.18668, 10.3660),
            ((1024, 41, 41), 3, 0.19061, 10.1504),
            ((32, 13, 11), 5, 0.11180, 6.3724),
            ((64, 48, 31), 1, 0.18668, 10.3660),
        ):
            target = {"lambda_": bandwidth_ratio, "gamma": gamma}
            _, plotted = validation_model_values(target, list(tile), depth)
            expected = model_score(tile, depth, bandwidth_ratio, gamma)
            self.assertAlmostEqual(plotted, expected, places=12)

    def test_aligned_full_width(self):
        sectors, edges = final_store_factors(1024, 9)
        self.assertEqual(sectors, 1.0)
        self.assertEqual(edges, 0.0)
        target = {"lambda_": 0.18668, "gamma": 10.3660}
        ideal, calibrated = validation_model_values(target, [96, 82, 74], 9)
        self.assertEqual(ideal, calibrated)
