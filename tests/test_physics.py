"""Conservation, analytic-gradient, and published-result regression checks."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from wpt.config import CFG
from wpt.model import Aperture
from wpt.optimization import PhaseOptimizer
from wpt.paths import CACHE_DIR, RESULTS_DIR


class PhysicsTests(unittest.TestCase):
    def test_power_conservation_and_slice(self):
        grid = Aperture(n=128)
        spectrum = grid.source(0.10, 3.0)
        for distance in [0.5, 3.0]:
            plane = grid.plane(spectrum, distance)
            self.assertAlmostEqual(float(plane["sz"].sum() * grid.dx**2), 1.0, places=12)
            x, flux, electric, magnetic = grid.longitudinal(spectrum, [distance])
            selected = np.abs(grid.x) <= 0.75
            np.testing.assert_array_equal(x, grid.x[selected])
            for actual, key in [(flux, "sz"), (electric, "e2"), (magnetic, "h2")]:
                np.testing.assert_allclose(
                    actual[0], plane[key][grid.n // 2, selected], rtol=2e-11, atol=1e-11,
                )

    def test_adjoint_gradient(self):
        optimizer = PhaseOptimizer(Aperture(n=128), 0.20, 3.0)
        rng = np.random.default_rng(42)
        phase = optimizer.initial + rng.normal(0, 0.05, optimizer.initial.size)
        direction = rng.normal(size=phase.size)
        direction /= np.linalg.norm(direction)
        _, gradient = optimizer.evaluate(phase, gradient=True)
        step = 1e-5
        finite_difference = (
            optimizer.evaluate(phase + step * direction)
            - optimizer.evaluate(phase - step * direction)
        ) / (2 * step)
        self.assertAlmostEqual(float(gradient @ direction), finite_difference, delta=1e-8)

    def test_configured_channels_and_common_phase(self):
        config = copy.deepcopy(CFG)
        config["tx_channels_per_axis"] = 8
        config["phase_bits"] = 4
        grid = Aperture(n=128, config=config)
        optimizer = PhaseOptimizer(grid, 0.10, 3.0)
        self.assertEqual(optimizer.initial.size, 64)
        step = 2 * np.pi / 2**config["phase_bits"]
        phase = np.round(optimizer.initial / step) * step
        expected = optimizer.spectrum(phase)
        np.testing.assert_allclose(grid.source(0.10, 3.0, bits=4), expected, atol=1e-11)
        original = grid.plane(expected, 3.0)
        shifted = grid.plane(optimizer.spectrum(phase + 0.37), 3.0)
        for key in original:
            np.testing.assert_allclose(original[key], shifted[key], rtol=2e-11, atol=1e-10)

    def test_saved_30cm_result(self):
        saved = json.loads((RESULTS_DIR / "optimized_summary.json").read_text(encoding="utf-8"))
        baseline = json.loads((RESULTS_DIR / "summary.json").read_text(encoding="utf-8"))
        grid = Aperture(config=baseline["config"])
        phase = np.array(saved["phase_radians_12_by_12_flat"]["0.3"])
        optimizer = PhaseOptimizer(grid, 0.30, 3.0)
        actual = grid.summarize(grid.plane(optimizer.spectrum(phase), 3.0))
        expected = saved["best_sampled"]
        for key in ["capture_fraction", "rf_feed_for_5w_w", "radiated_for_5w_w", "wall_for_5w_w"]:
            self.assertAlmostEqual(actual[key], expected[key], delta=max(1e-10, abs(expected[key])*1e-11))
        self.assertAlmostEqual(actual["plane_power_w_per_radiated_w"], 1.0, places=12)

    def test_derived_analysis_matches_saved_results(self):
        from wpt.analysis import run

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="test-", dir=CACHE_DIR) as temporary:
            target = Path(temporary)
            # The API reads and writes inside one result directory. Copy only
            # its small input set; all writes stay away from published results.
            import shutil

            for pattern in ["summary.json", "optimized_summary.json", "config.json", "optimized_fields_*.npz"]:
                for source in RESULTS_DIR.glob(pattern):
                    shutil.copyfile(source, target / source.name)
            run(output_dir=target)
            for name in ["public_point_screening.json", "receiver_sensitivity.json"]:
                actual = json.loads((target / name).read_text(encoding="utf-8"))
                expected = json.loads((RESULTS_DIR / name).read_text(encoding="utf-8"))
                self._assert_nested_close(actual, expected)

    def _assert_nested_close(self, actual, expected):
        if isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                self._assert_nested_close(actual[key], expected[key])
        elif isinstance(expected, list):
            self.assertEqual(len(actual), len(expected))
            for first, second in zip(actual, expected):
                self._assert_nested_close(first, second)
        elif isinstance(expected, (float, int)):
            self.assertAlmostEqual(actual, expected, delta=max(1e-10, abs(expected)*1e-11))
        else:
            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
