"""Checks for FEM input handling and isotropic energy invariants (no torch)."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluate_kratos import Settings, load_samples, nodal_average, resultant_mf, support_mask


class KratosEvaluationTests(unittest.TestCase):
    def test_physical_heights_preserved_and_singleton_batch(self):
        z = np.linspace(-.2, 5., 12).reshape(1, 1, 3, 4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.npz"
            np.savez(path, z=z)
            result = load_samples(path)
        self.assertEqual(result.shape, (1, 3, 4))
        np.testing.assert_array_equal(result, z[:, 0])

    def test_supports_use_ten_percent_of_max_without_recentering(self):
        z = np.array([[-.2, .5, .51], [0., 2., 5.], [.1, .2, 3.]])
        np.testing.assert_array_equal(support_mask(z, .1), z <= .5)

    def test_invalid_supports_rejected(self):
        for z in (np.ones((3, 3)), np.zeros((3, 3)), np.full((3, 3), np.nan)):
            with self.assertRaises(ValueError):
                support_mask(z, .1)

    def test_energy_limiting_cases_and_thickness(self):
        force = np.array([2., 3., 4.])
        zero = np.zeros(3)
        self.assertEqual(float(resultant_mf(force, zero, Settings())), 1.)
        self.assertEqual(float(resultant_mf(zero, force, Settings())), 0.)
        thin = float(resultant_mf(force, force, Settings(thickness=.1)))
        thick = float(resultant_mf(force, force, Settings(thickness=.2)))
        self.assertAlmostEqual(thin, 1/(1+12/.1**2))
        self.assertGreater(thick, thin)

    def test_resultant_mf_invariant_to_local_frame_rotation(self):
        theta = .67
        r = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        n, m = np.array([[2., 4.], [4., 3.]]), np.array([[.2, -.1], [-.1, .5]])
        def voigt(a):
            return np.array([a[0, 0], a[1, 1], a[0, 1]])
        self.assertAlmostEqual(float(resultant_mf(voigt(n), voigt(m), Settings())),
                               float(resultant_mf(voigt(r@n@r.T), voigt(r@m@r.T), Settings())))

    def test_nodal_average_and_exclusion(self):
        values = np.array([[1., 0.], [1., 0.]])
        np.testing.assert_array_equal(nodal_average(values), [[1., .5, 0.]]*3)
        np.testing.assert_array_equal(nodal_average(values, values > 0), [[1., 1., 0.]]*3)


if __name__ == "__main__":
    unittest.main()
