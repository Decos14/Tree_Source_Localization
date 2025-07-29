import unittest
import numpy as np
import tempfile
import os
import copy
from tree_source_localization.Tree import Tree  # type: ignore

class TestTree(unittest.TestCase):

    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w+', suffix='csv')
        self.temp_file.write("""A,B,N,mu=1.0;sigma2=0.5
B,C,E,lambda=2.0
C,D,U,start=1.0;stop=3.0
D,E,P,lambda=3.0
E,F,N,mu=1.0;sigma2=0.1
""")
        self.temp_file.close()

        self.observers = ["C", "D", "F"]
        self.infection_times = {obs: 0.0 for obs in self.observers}
        self.tree = Tree(self.temp_file.name, copy.deepcopy(self.observers), copy.deepcopy(self.infection_times))

    def tearDown(self):
        os.unlink(self.temp_file.name)

    def test_tree_structure_and_parameters(self):
        self.assertEqual(set(self.tree.nodes), {"A", "B", "C", "D", "E", "F"})
        self.assertEqual(len(self.tree.edges.keys()), 5)
        self.assertEqual(set(self.tree.edges[edge].dist_type for edge in self.tree.edges.keys()),
                         {"N", "E", "U", "P"})

    def test_connection_tree_validity(self):
        for node, neighbors in self.tree.connection_tree.items():
            for neighbor in neighbors:
                self.assertIn(node, self.tree.connection_tree[neighbor])

    def test_build_A_matrix_dimensions(self):
        A = self.tree.A
        self.assertEqual(set(A.keys()), set(self.tree.nodes))
        for node in A:
            self.assertEqual(A[node].shape, (len(self.observers), len(self.tree.edges.keys())))

    def test_edge_simulation_values(self):
        self.tree.simulate()
        for edge in self.tree.edges:
            val = self.tree.edges[edge].delay
            self.assertIsInstance(val, (float, int))
            self.assertGreaterEqual(val, 0)

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
        outfile = "test_equiv_class_tree.csv"
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

    def test_path_and_search_output_validity(self):
        for obs in self.observers:
            edges = self.tree.search.get_path("A", obs)
            for edge in edges:
                self.assertIsInstance(edge, frozenset)
                self.assertIn(edge, self.tree.edges.keys())


if __name__ == '__main__':
    unittest.main()
