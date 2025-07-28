# test_edge_distribution.py

import unittest
from tree_source_localization.EdgeDistribution import (
    EdgeDistribution,
    PositiveNormalDistribution,
    ExponentialDistribution,
    UniformDistribution,
    PoissonDistribution,
    AbsoluteCauchyDistribution
)


class TestEdgeDistribution(unittest.TestCase):

    def test_registry_resolves_correct_types(self):
        dists = {
            "N": {"mu": 0.0, "sigma2": 1.0},
            "E": {"lambda": 1.0},
            "U": {"start": 0.0, "stop": 1.0},
            "P": {"lambda": 1.0},
            "C": {"sigma2": 1.0},
        }

        expected_types = {
            "N": PositiveNormalDistribution,
            "E": ExponentialDistribution,
            "U": UniformDistribution,
            "P": PoissonDistribution,
            "C": AbsoluteCauchyDistribution,
        }

        for name, params in dists.items():
            inst = EdgeDistribution(name, params)
            self.assertIsInstance(inst.impl, expected_types[name])

    def test_distribution_instances_are_unique(self):
        dist1 = EdgeDistribution("E", {"lambda": 2.0})
        dist2 = EdgeDistribution("E", {"lambda": 2.0})

        self.assertIsNot(dist1.impl, dist2.impl)
        self.assertEqual(dist1.impl.params, dist2.impl.params)

    def test_sampling_and_mgfs(self):
        dist = EdgeDistribution("N", {"mu": 1.0, "sigma2": 1.0})
        dist.sample()
        sample = dist.delay
        self.assertIsInstance(sample, float)

        self.assertIsInstance(dist.mgf(0.1), float)
        self.assertIsInstance(dist.mgf_derivative(0.1), float)
        self.assertIsInstance(dist.mgf_derivative2(0.1), float)

    def test_registry_errors_on_unknown_distribution(self):
        with self.assertRaises(ValueError):
            EdgeDistribution("Z", {"foo": 1.0})


if __name__ == '__main__':
    unittest.main()