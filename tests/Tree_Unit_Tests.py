import unittest
import numpy as np
import tempfile
import os
import copy
import json
from tree_source_localization.Tree import Tree  # type: ignore

class TestTree(unittest.TestCase):

    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w+', suffix='json')
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
        json.dump(raw_edges, self.temp_file)
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
        self.tree.simulate_infection("A")
        for obs in self.observers:
            self.assertIn(obs, self.tree.infection_times)
            self.assertIsInstance(self.tree.infection_times[obs], (float, int))

    def test_joint_mgf_computation(self):
        self.tree.simulate()
        self.tree.simulate_infection("A")
        u = np.random.rand(len(self.observers))
        val = self.tree.joint_mgf(u, "A")
        self.assertIsInstance(val, float)
        self.assertGreater(val, 0)

    def test_cond_joint_mgf_methods(self):
        self.tree.simulate()
        self.tree.simulate_infection("A")
        u = np.random.rand(len(self.observers))
        for method in ['linear', 'exponential']:
            val = self.tree.cond_joint_mgf(u, "A", self.observers[0], method)
            self.assertIsInstance(val, float)

    def test_equivalent_class_output_validity(self):
        outfile = "test_equiv_class_tree.csv"
        self.tree.get_equivalent_class(self.observers[0], outfile)
        self.assertTrue(os.path.exists(outfile))
        os.remove(outfile)

    def test_obj_func_output_validity(self):
        self.tree.simulate()
        self.tree.simulate_infection("A")
        u = np.random.rand(len(self.observers))
        for method in [None, 'linear', 'exponential']:
            val = self.tree.objective_function(u, "A", method=method)
            self.assertIsInstance(val, float)

    def test_localize_returns_node(self):
        self.tree.simulate()
        self.tree.simulate_infection("A")
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
