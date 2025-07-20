import unittest
import pickle
import numpy as np
import copy
import tempfile
import os
from tree_source_localization.Tree import Tree # type: ignore



class TestTreeRegression(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open("regression_test_data.pkl", "rb") as file:
            cls.saved_results = pickle.load(file)

        cls.temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w+', suffix = 'csv')
        cls.temp_file.write("""A,B,N,1.0,0.5
B,C,E,2.0
C,D,U,1.0,3.0
D,E,P,3.0
E,F,C,1.0
""")
        cls.temp_file.close()

        cls.observers = ["C", "D", "F"]
        cls.infection_times = {obs: 0.0 for obs in cls.observers}

        cls.tree_new = Tree(cls.temp_file.name, copy.deepcopy(cls.observers), copy.deepcopy(cls.infection_times))
        cls.tree_new.build_tree(cls.temp_file.name)
        cls.tree_new.build_connection_tree()

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.temp_file.name)

    def test_edges_match(self):
        self.assertEqual(self.tree_new.edges, self.saved_results.get('edges'))

    def test_nodes_match(self):
        self.assertEqual(set(self.tree_new.nodes), set(self.saved_results.get('nodes')))

    def test_distributions_match(self):
        self.assertEqual(self.tree_new.distributions, self.saved_results.get('distributions'))

    def test_parameters_match(self):
        self.assertEqual(self.tree_new.parameters, self.saved_results.get('parameters'))

    def test_edge_delays_match(self):
        np.random.seed(42)
        self.tree_new.simulate()
        saved = self.saved_results.get('edge_delays')
        for edge in saved:
            self.assertAlmostEqual(self.tree_new.edge_delays[edge], saved[edge], places=5)

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
        self.tree_new.Infection_Simulation("A")
        saved_inf = self.saved_results.get("Infection_times")
        self.assertEqual(set(self.tree_new.infection_times.keys()), set(saved_inf.keys()))
        for k in saved_inf:
            self.assertAlmostEqual(self.tree_new.infection_times[k], saved_inf[k], places=5)

    def test_Joint_MGF_match(self):
        np.random.seed(42)
        self.tree_new.simulate()
        np.random.seed(42)
        self.tree_new.Infection_Simulation("A")

        np.random.seed(42)
        u = np.random.rand(len(self.observers))

        val = self.tree_new.joint_mgf(u, "A")
        saved_val = self.saved_results.get("Joint_MGF")
        if isinstance(val, np.ndarray):
            np.testing.assert_allclose(val, saved_val, rtol=1e-5, atol=1e-8)
        else:
            self.assertAlmostEqual(val, saved_val, places=5)

    def test_cond_joint_mgf_methods_match(self):
        np.random.seed(42)
        self.tree_new.simulate()
        np.random.seed(42)
        self.tree_new.Infection_Simulation("A")

        u_seed = 12345
        rng = np.random.default_rng(u_seed)
        u = rng.random(len(self.observers))

        for method in [1, 2]:
            val = self.tree_new.cond_joint_mgf(u, "A", self.observers[0], method)
            saved_val = self.saved_results.get(f"Cond_Joint_MGF_{method}")
            if isinstance(val, np.ndarray):
                val = float(val)
            if isinstance(saved_val, np.ndarray):
                saved_val = float(saved_val)
            if np.isnan(val) and np.isnan(saved_val):
                continue
            self.assertAlmostEqual(val, saved_val, places=3)

    def test_cond_joint_mgf_exp_approx_match(self):
        temp_file_exp = tempfile.NamedTemporaryFile(delete=False, mode='w+')
        temp_file_exp.write("A,B,E,2.0\nB,C,E,2.0\n")
        temp_file_exp.close()
        obs = ["C"]
        times = {"C": 0.0}
        tree_new = Tree(temp_file_exp.name, obs, times)
        tree_new.build_tree(temp_file_exp.name)

        np.random.seed(42)
        tree_new.simulate()
        np.random.seed(42)
        tree_new.Infection_Simulation("A")

        u_seed = 12345
        rng = np.random.default_rng(u_seed)
        u = rng.random(len(obs))

        val = tree_new.cond_joint_mgf(u, "A", "C", 3)
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
        self.tree_new.Infection_Simulation("A")

        u_seed = 12345
        rng = np.random.default_rng(u_seed)
        u = rng.random(len(self.observers))

        saved_val = self.saved_results.get("Objective_Function")
        for aug in [None, 1, 2]:
            val = self.tree_new.obj_func(u, "A", augment=aug)
            if np.isnan(val) and np.isnan(saved_val):
                continue
            self.assertAlmostEqual(val, saved_val, places=5)

    def test_localize_output_match(self):
        np.random.seed(42)
        self.tree_new.simulate()
        np.random.seed(42)
        self.tree_new.Infection_Simulation("A")

        saved_val = self.saved_results.get("localize")
        val = self.tree_new.localize()
        self.assertIsInstance(val, str)
        self.assertEqual(val, saved_val)


if __name__ == '__main__':
    unittest.main()
