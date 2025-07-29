import unittest
import pickle
import numpy as np
import copy
import json
import tempfile
import os
from tree_source_localization.Tree import Tree  # type: ignore

REGRESSION_PATH = "tree_regression_test_data.pkl"

class TestTreeRegression(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w+', suffix='json')
        raw_edges = {
            "A,B" : {
                'distribution': 'N',
                'parameters': {
                    'mu': 1.0,
                    'sigma2': 0.5
                }
            },
            "B,C" : {
                'distribution': 'E',
                'parameters': {
                    'lambda': 2.0
                }
            },
            "C,D" : {
                'distribution': 'U',
                'parameters': {
                    'start': 1.0,
                    'stop': 3.0
                }
            },
            "D,E" : {
                'distribution': 'P',
                'parameters': {
                    'lambda': 3.0
                }
            },
            "E,F" : {
                'distribution': 'N',
                'parameters': {
                    'mu': 1.0,
                    'sigma2': 0.1
                }
            }
        }
        json.dump(raw_edges, cls.temp_file)
        cls.temp_file.close()

        cls.observers = ["C", "D", "F"]
        cls.infection_times = {obs: 0.0 for obs in cls.observers}

        cls.tree_new = Tree(cls.temp_file.name, copy.deepcopy(cls.observers), copy.deepcopy(cls.infection_times))

        if not os.path.exists(REGRESSION_PATH):
            np.random.seed(42)
            cls.tree_new.simulate()
            np.random.seed(42)
            cls.tree_new.simulate_infection("A")
            cls.tree_new.build_A_matrix()
            np.random.seed(42)
            u = np.random.rand(len(cls.observers))
            temp_file_exp = tempfile.NamedTemporaryFile(delete=False, mode='w+')
            temp_file_exp.write("A,B,E,lambda=2.0\nB,C,E,lambda=2.0\n")
            temp_file_exp.close()
            obs = ["C"]
            times = {"C": 0.0}
            tree_new_3 = Tree(temp_file_exp.name, obs, times)

            np.random.seed(42)
            tree_new_3.simulate()
            np.random.seed(42)
            tree_new_3.simulate_infection("A")

            np.random.seed(42)
            u_3 = np.random.rand((len(obs)))

            cond_mgf_3_val = tree_new_3.cond_joint_mgf(u_3, "A", "C", 3)

            # Save data
            data = {
                "nodes": cls.tree_new.nodes,
                "distributions": {edge: cls.tree_new.edges[edge].dist_type for edge in cls.tree_new.edges},
                "parameters": {edge: cls.tree_new.edges[edge].params for edge in cls.tree_new.edges},
                "edge_delays": {edge: cls.tree_new.edges[edge].delay for edge in cls.tree_new.edges},
                "connection_tree": cls.tree_new.connection_tree,
                "A": cls.tree_new.A,
                "Infection_times": cls.tree_new.infection_times,
                "Joint_MGF": cls.tree_new.joint_mgf(u, "A"),
                "Cond_Joint_MGF_1": cls.tree_new.cond_joint_mgf(u, "A", cls.observers[0], 1),
                "Cond_Joint_MGF_2": cls.tree_new.cond_joint_mgf(u, "A", cls.observers[0], 2),
                "Cond_Joint_MGF_3": cond_mgf_3_val,
                "Objective_Function": cls.tree_new.objective_function(u, "A"),
                "localize": cls.tree_new.localize(),
            }
            with open(REGRESSION_PATH, "wb") as f:
                pickle.dump(data, f)

        with open(REGRESSION_PATH, "rb") as file:
            cls.saved_results = pickle.load(file)
    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.temp_file.name)

    def test_nodes_match(self):
        self.assertEqual(set(self.tree_new.nodes), set(self.saved_results.get('nodes')))

    def test_distributions_match(self):
        # Now distributions are EdgeDistribution instances keyed by edge
        dist_dict = {edge: self.tree_new.edges[edge].dist_type for edge in self.tree_new.edges}
        self.assertEqual(dist_dict, self.saved_results.get('distributions'))

    def test_parameters_match(self):
        param_dict = {edge: self.tree_new.edges[edge].params for edge in self.tree_new.edges}
        self.assertEqual(param_dict, self.saved_results.get('parameters'))

    def test_edge_delays_match(self):
        np.random.seed(42)
        self.tree_new.simulate()
        saved = self.saved_results.get('edge_delays')
        for edge in saved:
            self.assertAlmostEqual(self.tree_new.edges[edge].delay, saved[edge], places=5)

    def test_connection_tree_match(self):
        saved_conn = self.saved_results.get('connection_tree')
        self.assertEqual(set(self.tree_new.connection_tree.keys()), set(saved_conn.keys()))
        for k in saved_conn:
            self.assertCountEqual(self.tree_new.connection_tree[k], saved_conn[k])

    def test_A_matrix_match(self):
        saved_A = self.saved_results.get('A')
        self.tree_new.build_A_matrix()
        A_new = self.tree_new.A
        for node in saved_A:
            current_matrix = A_new[node]
            saved_matrix = saved_A[node]
            np.testing.assert_allclose(current_matrix, saved_matrix, rtol=1e-5, atol=1e-8)

    def test_Infection_times_match(self):
        np.random.seed(42)
        self.tree_new.simulate()
        np.random.seed(42)
        self.tree_new.simulate_infection("A")
        saved_inf = self.saved_results.get("Infection_times")
        self.assertEqual(set(self.tree_new.infection_times.keys()), set(saved_inf.keys()))
        for k in saved_inf:
            self.assertAlmostEqual(self.tree_new.infection_times[k], saved_inf[k], places=5)

    def test_Joint_MGF_match(self):
        np.random.seed(42)
        self.tree_new.simulate()
        np.random.seed(42)
        self.tree_new.simulate_infection("A")

        np.random.seed(42)
        u = np.random.rand(len(self.observers))

        val = self.tree_new.joint_mgf(u, "A")
        saved_val = self.saved_results.get("Joint_MGF")
        if isinstance(val, np.ndarray):
            np.testing.assert_allclose(val, saved_val, rtol=1e-5, atol=1e-8)
        else:
            self.assertAlmostEqual(val, saved_val, places=5)

    def test_cond_joint_mgf_method_1_match(self):
        np.random.seed(42)
        self.tree_new.simulate()
        np.random.seed(42)
        self.tree_new.simulate_infection("A")

        np.random.seed(42)
        u = np.random.rand(len(self.observers))

        val = self.tree_new.cond_joint_mgf(u, "A", self.observers[0], 'linear')
        saved_val = self.saved_results.get(f"Cond_Joint_MGF_1")
        if isinstance(val, np.ndarray):
            val = float(val)
        if isinstance(saved_val, np.ndarray):
            saved_val = float(saved_val)
        self.assertAlmostEqual(val, saved_val, places=3)
        
    def test_cond_joint_mgf_method_2_match(self):
        np.random.seed(42)
        self.tree_new.simulate()
        np.random.seed(42)
        self.tree_new.simulate_infection("A")

        np.random.seed(42)
        u = np.random.rand(len(self.observers))
        val = self.tree_new.cond_joint_mgf(u, "A", self.observers[0], 'exponential')
        saved_val = self.saved_results.get(f"Cond_Joint_MGF_2")
        if isinstance(val, np.ndarray):
            val = float(val)
        if isinstance(saved_val, np.ndarray):
            saved_val = float(saved_val)
        self.assertAlmostEqual(val, saved_val, places=3)

    def test_cond_joint_mgf_exp_approx_match(self):
        temp_file_exp = tempfile.NamedTemporaryFile(delete=False, mode='w+', suffix='json')
        raw_exp_edges = {
            "A,B" : {
                'distribution': 'E',
                'parameters': {
                    'lambda': 2.0
                }
            },
            "B,C" : {
                'distribution': 'E',
                'parameters': {
                    'lambda': 2.0
                }
            }
        }
        json.dump(raw_exp_edges, temp_file_exp)
        temp_file_exp.close()
        obs = ["C"]
        times = {"C": 0.0}
        tree_new = Tree(temp_file_exp.name, obs, times)

        np.random.seed(42)
        tree_new.simulate()
        np.random.seed(42)
        tree_new.simulate_infection("A")

        np.random.seed(42)
        u = np.random.rand(len(obs))

        val = tree_new.cond_joint_mgf(u, "A", "C", 'exact')
        saved_val = self.saved_results.get("Cond_Joint_MGF_3")

        os.unlink(temp_file_exp.name)

        if isinstance(val, np.ndarray):
            np.testing.assert_allclose(val, saved_val, rtol=1e-5, atol=1e-8)
        else:
            self.assertAlmostEqual(val, saved_val, places=5)

    def test_obj_func_match(self):
        np.random.seed(42)
        self.tree_new.simulate()
        np.random.seed(42)
        self.tree_new.simulate_infection("A")

        np.random.seed(42)
        u = np.random.rand(len(self.observers))

        saved_val = self.saved_results.get("Objective_Function")
        val = self.tree_new.objective_function(u, "A")
        self.assertAlmostEqual(val, saved_val, places=5)
    
    def test_obj_func_1_match(self):
        np.random.seed(42)
        self.tree_new.simulate()
        np.random.seed(42)
        self.tree_new.simulate_infection("A")

        np.random.seed(42)
        u = np.random.rand(len(self.observers))

        saved_val = self.saved_results.get("Objective_Function")
        val = self.tree_new.objective_function(u, "A", method='linear')
        self.assertAlmostEqual(val, saved_val, places=5)

    def test_obj_func_2_match(self):
        np.random.seed(42)
        self.tree_new.simulate()
        np.random.seed(42)
        self.tree_new.simulate_infection("A")

        np.random.seed(42)
        u = np.random.rand(len(self.observers))

        saved_val = self.saved_results.get("Objective_Function")
        val = self.tree_new.objective_function(u, "A", method='exponential')
        self.assertAlmostEqual(val, saved_val, places=5)

    def test_localize_output_match(self):
        np.random.seed(42)
        self.tree_new.simulate()
        np.random.seed(42)
        self.tree_new.simulate_infection("A")

        saved_val = self.saved_results.get("localize")
        val = self.tree_new.localize()
        self.assertIsInstance(val, str)
        self.assertEqual(val, saved_val)


if __name__ == '__main__':
    unittest.main()
