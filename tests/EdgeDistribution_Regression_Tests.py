import unittest
import pickle
import os
import numpy as np
from tree_source_localization.EdgeDistribution import EdgeDistribution

REGRESSION_PICKLE = "edge_distribution_regression_test_data.pkl"

class TestEdgeDistributionRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create example EdgeDistribution instances with fixed params
        cls.test_edges = {
            "N": EdgeDistribution("N", {"mu": 1.0, "sigma2": 0.5}),
            "E": EdgeDistribution("E", {"lambda": 2.0}),
            "U": EdgeDistribution("U", {"start": 1.0, "stop": 3.0}),
            "P": EdgeDistribution("P", {"lambda": 3.0}),
        }
        
        # Prepare to store regression data (samples, mgfs at some t)
        cls.regression_data = {}
        
        # Random seed for reproducibility
        np.random.seed(42)

        if os.path.exists(REGRESSION_PICKLE):
            with open(REGRESSION_PICKLE, "rb") as f:
                cls.regression_data = pickle.load(f)
        else:
            # Generate regression data and save it
            for dist_key, ed in cls.test_edges.items():
                # Generate samples
                samples = [ed.sample() for _ in range(10)]
                
                # Evaluate mgf and derivatives at a few points t
                ts = [0, 0.1, 1.0]
                mgf_vals = [ed.mgf(t) for t in ts]
                mgf_deriv_vals = [ed.mgf_derivative(t) for t in ts]
                mgf_deriv2_vals = [ed.mgf_derivative2(t) for t in ts]
                
                cls.regression_data[dist_key] = {
                    "samples": samples,
                    "mgf": mgf_vals,
                    "mgf_derivative": mgf_deriv_vals,
                    "mgf_derivative2": mgf_deriv2_vals,
                }
            
            with open(REGRESSION_PICKLE, "wb") as f:
                pickle.dump(cls.regression_data, f)
            print(f"Created regression pickle file '{REGRESSION_PICKLE}'.")

    def test_samples_regression(self):
        for dist_key, ed in self.test_edges.items():
            expected_samples = self.regression_data[dist_key]["samples"]
            # Re-seed to get same randomness order
            np.random.seed(42)
            actual_samples = [ed.sample() for _ in range(10)]
            for exp_val, act_val in zip(expected_samples, actual_samples):
                self.assertAlmostEqual(exp_val, act_val, places=5, msg=f"Sample mismatch in {dist_key}")

    def test_mgf_regression(self):
        ts = [0, 0.1, 1.0]
        for dist_key, ed in self.test_edges.items():
            expected_vals = self.regression_data[dist_key]["mgf"]
            for t, expected in zip(ts, expected_vals):
                actual = ed.mgf(t)
                self.assertAlmostEqual(expected, actual, places=5, msg=f"MGF mismatch for {dist_key} at t={t}")

    def test_mgf_derivative_regression(self):
        ts = [0, 0.1, 1.0]
        for dist_key, ed in self.test_edges.items():
            expected_vals = self.regression_data[dist_key]["mgf_derivative"]
            for t, expected in zip(ts, expected_vals):
                actual = ed.mgf_derivative(t)
                self.assertAlmostEqual(expected, actual, places=5, msg=f"MGF derivative mismatch for {dist_key} at t={t}")

    def test_mgf_derivative2_regression(self):
        ts = [0, 0.1, 1.0]
        for dist_key, ed in self.test_edges.items():
            expected_vals = self.regression_data[dist_key]["mgf_derivative2"]
            for t, expected in zip(ts, expected_vals):
                actual = ed.mgf_derivative2(t)
                self.assertAlmostEqual(expected, actual, places=5, msg=f"MGF 2nd derivative mismatch for {dist_key} at t={t}")

if __name__ == "__main__":
    unittest.main()
