import unittest
import numpy as np
import tempfile
import os
import copy
import sys
sys.path.append(r'C:\Users\devli\Documents\Research\Source Localization\New Functions')
from Tree_refactor import Tree as NewTree
from Tree import Tree as OriginalTree  # Rename accordingly

class TestTree(unittest.TestCase):

    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w+')
        self.temp_file.write("""A,B,N,1.0,0.5
B,C,E,2.0
C,D,U,1.0,3.0
D,E,P,3.0
E,F,C,1.0
""")
        self.temp_file.close()

        self.observers = ["C", "D", "F"]
        self.infection_times = {obs: 0.0 for obs in self.observers}

        self.tree_new = NewTree(self.temp_file.name, copy.deepcopy(self.observers), copy.deepcopy(self.infection_times))
        self.tree_orig = OriginalTree(self.temp_file.name, copy.deepcopy(self.observers), copy.deepcopy(self.infection_times))

    def tearDown(self):
        os.unlink(self.temp_file.name)

    def test_build_tree_structure_and_content(self):
        self.tree_new.build_tree(self.temp_file.name)
        tree_new = self.tree_new.tree
        dist_new = self.tree_new.distributions
        tree_orig = self.tree_orig.build_tree(self.temp_file.name)
        self.assertEqual(tree_new.keys(), tree_orig.keys())
        for edge in tree_new:
            self.assertEqual(dist_new[edge], tree_orig[edge][0][0])

    def test_connection_tree_consistency(self):
        self.tree_new.build_connection_tree()
        conn_new = self.tree_new.connection_tree
        conn_orig = self.tree_orig.build_connection_tree(self.tree_orig.tree)
        self.assertEqual(conn_new.keys(), conn_orig.keys())
        for node in conn_new:
            self.assertCountEqual(conn_new[node], conn_orig[node])

    def test_build_A_matrix_values(self):
        A_new = self.tree_new.build_A_matrix()  # dict: node -> (obs x edges) matrix
        A_orig = self.tree_orig.build_A_matrix()  # shape: (nodes x obs x edges)
        self.assertEqual(len(A_new), A_orig.shape[0])  # same number of node matrices

        new_values = list(A_new.values())

        for i in range(A_orig.shape[0]):
            orig_slice = A_orig[i]
            match_found = any(np.array_equal(orig_slice, new_matrix) for new_matrix in new_values)
            self.assertTrue(match_found, f"No match found for A_orig[{i}] in A_new values.")



    def test_simulate_edge_distribution_outputs(self):
        for edge, val in self.tree_new.tree.items():
            d = self.tree_new.simulate_edge(edge)
            self.assertIsInstance(d, (float, int))
            self.assertGreaterEqual(d, 0)

    def test_simulate_sets_delay(self):
        self.tree_new.simulate()
        for edge in self.tree_new.tree:
            self.assertIsInstance(self.tree_new.edge_delays[edge], (float, int))

    def test_DFS_path_correctness(self):
        for observer in self.observers:
            path = self.tree_new.DFS("A", observer)
            self.assertTrue(path[0] == "A" and path[-1] == observer)
            edges = self.tree_new.path_edge(path)
            self.assertTrue(all(isinstance(e, frozenset) for e in edges))

    def test_Infection_Simulation_matches_original(self):
        np.random.seed(42)
        self.tree_new.simulate()
        np.random.seed(42)
        self.tree_orig.simulate()
        self.tree_new.Infection_Simulation("A")
        self.tree_orig.Infection_Simulation("A")
        self.assertEqual(self.tree_new.infection_times.keys(), self.tree_orig.infection_times.keys())
        for key in self.tree_new.infection_times:
            self.assertAlmostEqual(self.tree_new.infection_times[key], self.tree_orig.infection_times[key], places=5)

    def test_joint_mgf_matches_original(self):
        np.random.seed(42)
        self.tree_new.simulate()
        np.random.seed(42)
        self.tree_orig.simulate()
        self.tree_new.Infection_Simulation("A")
        self.tree_orig.Infection_Simulation("A")
        u = np.random.rand(len(self.observers))
        val_new = self.tree_new.joint_mgf(u, "A")
        val_orig = self.tree_orig.joint_mgf(u, "A")
        if isinstance(val_new, np.ndarray):
            np.testing.assert_allclose(val_new, val_orig, rtol=1e-5, atol=1e-8)
        else:
            self.assertAlmostEqual(val_new, val_orig, places=5)

    def test_cond_joint_mgf_all_methods_match(self):
        np.random.seed(42)
        self.tree_new.simulate()
        np.random.seed(42)
        self.tree_orig.simulate()
        self.tree_new.Infection_Simulation("A")
        self.tree_orig.Infection_Simulation("A")
        u = np.random.rand(len(self.observers))
        for method in [1, 2]:
            val_new = self.tree_new.cond_joint_mgf(u, "A", self.observers[0], method)
            val_orig = self.tree_orig.cond_joint_mgf(u, "A", self.observers[0], method)
            if isinstance(val_new, np.ndarray):
                val_new = float(val_new)
            if isinstance(val_orig, np.ndarray):
                val_orig = float(val_orig)
            # If both are NaN, consider them equal for the test
            if np.isnan(val_new) and np.isnan(val_orig):
                continue
            self.assertAlmostEqual(val_new, val_orig, places=3)

    def test_cond_joint_mgf_exp_approx(self):
        temp_file_exp = tempfile.NamedTemporaryFile(delete=False, mode='w+')
        temp_file_exp.write("A,B,E,2.0\nB,C,E,2.0\n")
        temp_file_exp.close()
        obs = ["C"]
        times = {"C": 0.0}
        tree_new = NewTree(temp_file_exp.name, obs, times)
        tree_orig = OriginalTree(temp_file_exp.name, obs, times)
        np.random.seed(42)
        tree_new.simulate()
        np.random.seed(42)
        tree_orig.simulate()
        tree_new.Infection_Simulation("A")
        tree_orig.Infection_Simulation("A")
        u = np.random.rand(len(obs))
        val_new = tree_new.cond_joint_mgf(u, "A", "C", 3)
        val_orig = tree_orig.cond_joint_mgf(u, "A", "C", 3)
        if isinstance(val_new, np.ndarray):
            np.testing.assert_allclose(val_new, val_orig, rtol=1e-5, atol=1e-8)
        else:
            self.assertAlmostEqual(val_new, val_orig, places=5)
        os.unlink(temp_file_exp.name)

    def test_Equivalent_Class_structure_and_match(self):
        outfile = "test_equivalence_class.txt"
        new_obs =  self.tree_new.Equivalent_Class(self.observers[0], outfile)
        result_new = NewTree(outfile, new_obs, self.infection_times)
        os.remove(outfile)
        result_orig = self.tree_orig.Equivalent_Class(self.observers[0])
        self.assertEqual(result_new.edges, list(result_orig.keys()))

    def test_obj_func_behavior_and_match(self):
        np.random.seed(42)
        self.tree_new.simulate()
        np.random.seed(42)
        self.tree_orig.simulate()
        self.tree_new.Infection_Simulation("A")
        self.tree_orig.Infection_Simulation("A")
        u = np.random.rand(len(self.observers))
        for aug in [None, 1, 2]:
            val_new = self.tree_new.obj_func(u, "A", augment=aug)
            val_orig = self.tree_orig.obj_func(u, "A", augment=aug)
            if np.isnan(val_new) and np.isnan(val_orig):
                continue
            self.assertAlmostEqual(val_new, val_orig, places=5)

    def test_localize_output_type_and_differs(self):
        np.random.seed(42)
        self.tree_new.simulate()
        np.random.seed(42)
        self.tree_orig.simulate()
        self.tree_new.Infection_Simulation("A")
        self.tree_orig.Infection_Simulation("A")
        loc_new = self.tree_new.localize()
        loc_orig = self.tree_orig.localize()
        self.assertIsInstance(loc_new, str)
        self.assertIsInstance(loc_orig, str)

if __name__ == '__main__':
    unittest.main()