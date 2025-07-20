import unittest
import numpy as np
import tempfile
import os
import copy
from tree_source_localization.Tree import Tree



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
        self.tree = Tree(self.temp_file.name, copy.deepcopy(self.observers), copy.deepcopy(self.infection_times))

    def tearDown(self):
        os.unlink(self.temp_file.name)

    def test_tree_structure_and_parameters(self):
        self.assertEqual(set(self.tree.nodes), {"A", "B", "C", "D", "E", "F"})
        self.assertEqual(len(self.tree.edges), 5)
        self.assertIn("N", set(self.tree.distributions.values()))
        self.assertIn("E", set(self.tree.distributions.values()))
        self.assertIn("U", set(self.tree.distributions.values()))
        self.assertIn("P", set(self.tree.distributions.values()))
        self.assertIn("C", set(self.tree.distributions.values()))

    def test_connection_tree_validity(self):
        for node, neighbors in self.tree.connection_tree.items():
            for neighbor in neighbors:
                self.assertIn(node, self.tree.connection_tree[neighbor])

    def test_build_A_matrix_dimensions(self):
        A = self.tree.A
        self.assertEqual(set(A.keys()), set(self.tree.nodes))
        for node in A:
            self.assertEqual(A[node].shape, (len(self.observers), len(self.tree.edges)))

    def test_edge_simulation_per_distribution(self):
        dist_to_sampler = {
            'N': lambda: Tree._simulate_edge(self.tree, frozenset({"X", "Y"})),
            'E': lambda: Tree._simulate_edge(self.tree, frozenset({"X", "Y"})),
            'U': lambda: Tree._simulate_edge(self.tree, frozenset({"X", "Y"})),
            'P': lambda: Tree._simulate_edge(self.tree, frozenset({"X", "Y"})),
            'C': lambda: Tree._simulate_edge(self.tree, frozenset({"X", "Y"})),
        }
        # Just check all actual edges simulate positive numbers
        self.tree.simulate()
        for edge in self.tree.edges:
            d = self.tree.edge_delays[edge]
            self.assertIsInstance(d, (int, float))
            self.assertGreaterEqual(d, 0)

    def test_infection_simulation_sets_times(self):
        self.tree.simulate()
        self.tree.Infection_Simulation("A")
        for obs in self.observers:
            self.assertIn(obs, self.tree.infection_times)
            self.assertIsInstance(self.tree.infection_times[obs], (float, int))

    def test_joint_mgf_computation(self):
        self.tree.simulate()
        self.tree.Infection_Simulation("A")
        u = np.random.rand(len(self.observers))
        val = self.tree.joint_mgf(u, "A")
        self.assertIsInstance(val, float)
        self.assertGreater(val, 0)

    def test_cond_joint_mgf_methods(self):
        self.tree.simulate()
        self.tree.Infection_Simulation("A")
        u = np.random.rand(len(self.observers))
        for method in [1, 2]:
            val = self.tree.cond_joint_mgf(u, "A", self.observers[0], method)
            self.assertIsInstance(val, float)

    def test_equivalent_class_output_validity(self):
        outfile = "test_equiv_class_tree.txt"
        new_obs = self.tree.Equivalent_Class(self.observers[0], outfile)
        self.assertTrue(set(new_obs).issubset(set(self.observers)))
        self.assertTrue(os.path.exists(outfile))
        os.remove(outfile)

    def test_obj_func_output_validity(self):
        self.tree.simulate()
        self.tree.Infection_Simulation("A")
        u = np.random.rand(len(self.observers))
        for method in [None, 1, 2]:
            val = self.tree.obj_func(u, "A", augment=method)
            self.assertIsInstance(val, float)

    def test_localize_returns_node(self):
        self.tree.simulate()
        self.tree.Infection_Simulation("A")
        loc = self.tree.localize()
        self.assertIn(loc, self.tree.nodes)

    def test_path_and_DFS_output_validity(self):
        for obs in self.observers:
            path = self.tree.DFS("A", obs)
            self.assertEqual(path[0], "A")
            self.assertEqual(path[-1], obs)
            edges = self.tree.path_edge(path)
            for edge in edges:
                self.assertIsInstance(edge, frozenset)
                self.assertIn(edge, self.tree.edges)

if __name__ == '__main__':
    unittest.main()
