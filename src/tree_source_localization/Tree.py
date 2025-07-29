import numpy as np
import scipy as sp
import json
from typing import List, Set, FrozenSet, Dict
from numpy.typing import ArrayLike
from .Search import _DepthFirstSearch
from .MGFAugment import get_augmentation
from .EdgeDistribution import EdgeDistribution

#Structure is {node, node}
TreeEdge = FrozenSet[str]

class Tree:
    def __init__(
        self,
        file_name: str,
        observers: List[str],
        infection_times: Dict[int, str]
    ) -> None:
        self.file_name = file_name
        self.build_tree(file_name)
        self.build_connection_tree()
        self.infection_times = infection_times
        self.observers = observers
        self.search = _DepthFirstSearch(self.connection_tree)
        self.build_A_matrix()

    def build_tree(
        self, 
        file_name: str
    ) -> None:
        """
        Builds the tree data structure from a JSON file, initializing edges, nodes,
        distributions, parameters, delays, and moment generating functions as instance variables.

        Args:
            file_name (str): Path to the JSON file describing the tree.
        """
        with open(file_name, 'r', encoding='utf-8') as file:
            raw_edges = json.load(file)
        
        self.edges = {
            frozenset(edge.split(',')) : EdgeDistribution(value['distribution'], value['parameters'])
            for edge, value in raw_edges.items()
        }

        self.nodes = list(set().union(*list(self.edges.keys())))
    
    def build_connection_tree(self) -> None:
        """
        Builds an adjacency dictionary representing the tree topology and assigns it
        to the instance variable `self.connection_tree`.
        """
        connection_tree = {node: [] for node in self.nodes}
        for edge in self.edges:
            u, v = tuple(edge)
            connection_tree[u].append(v)
            connection_tree[v].append(u)
        self.connection_tree = connection_tree
    
    def build_A_matrix(self) -> None:
        """
        Constructs the A-matrix tensor of the tree using the observers and assigns it
        to the instance variable `self.A`.
        """
        A_matrix = {}
        for node in self.nodes:
            A_layer= np.zeros((len(self.observers),len(self.edges.keys())))
            for j, obs in enumerate(self.observers):
                path = self.search.get_path(node,obs)
                for k, edge in enumerate(self.edges.keys()):
                    if edge in path:
                        A_layer[j,k]=1  
            A_matrix[node] = A_layer 
        self.A = A_matrix

    def simulate(self) -> None:
        """
        Simulates delay values for all edges in the tree and updates the instance
        variable `self.edge_delays` with these simulated values.
        """
        for edge in self.edges:
            self.edges[edge].sample()
    
    def simulate_infection(
        self,
        source: str
    ) -> None:
        """
        Simulates the infection spread times from a source node to all observers and
        stores the results in the instance variable `self.infection_times`.

        Args:
            source (str): The source node from which the infection starts.
        """
        infection_times = {}

        for observer in self.observers:
            edges = self.search.get_path(source, observer)
            time = 0
            for edge in edges:
                time += self.edges[edge].delay
            infection_times[observer] = time
        self.infection_times= infection_times

    def joint_mgf(
        self,
        u: ArrayLike,
        source: str
    ) -> float:
        """
        Computes the joint Moment Generating Function (MGF) of the infection times
        for the observers from a given source node, evaluated at vector `u`.

        Args:
            u (ArrayLike): Vector to evaluate the joint MGF at.
            source (str): Node assumed to be the infection source.

        Returns:
            float: The value of the joint MGF at `u`.
        """
        mgf = 1
        for i,edge in enumerate(self.edges.keys()):
            relevant_u = np.matmul(u,self.A[source][:,i])
            if relevant_u != 0:
                mgf *= self.edges[edge].mgf(relevant_u)
        return mgf
    
    def cond_joint_mgf(
        self,
        u: ArrayLike,
        source: str,
        obs_o: str,
        method: int
    ) -> float:
        """
        Computes or approximates the conditional joint MGF of the observers given the
        first infected observer, using a specified augmentation method.

        Args:
            u (ArrayLike): Vector to evaluate the conditional joint MGF at.
            source (str): Assumed infection source node.
            obs_o (str): The first observer infected.
            method (int): Augmentation method to use:
                1: Linear approximation,
                2: Exponential approximation,
                3: Exact solution for iid exponential delays.

        Returns:
            float: Conditional joint MGF evaluated at `u`.
        """
        mgf = 1

        path  = self.search.get_path(source, obs_o)
        for i,edge in enumerate(self.edges.keys()):
            if edge not in path:
                relevant_u = np.matmul(u,self.A[source][:,i])
                if relevant_u  != 0:
                    mgf *= self.edges[edge].mgf(relevant_u )
        
        augment = get_augmentation(method)
        mgf *= augment(u, self.A[source], self.infection_times[obs_o], path, self.edges)
        return mgf
    
    def _get_subtree_nodes(
        self,
        root: str
        ) -> Set[str]:
        """
        Performs a BFS traversal from the given root node to find all nodes in the
        connected subtree that are not separated by observer boundaries.

        Args:
            root (str): The starting node (typically the first observer infected).

        Returns:
            Set[str]: Set of nodes in the relevant subtree.
        """
        to_check = self.connection_tree[root][:]
        visited = {root}

        while to_check:
            current = to_check.pop(0)
            if current not in visited:
                visited.add(current)
                if current not in self.observers:
                    to_check.extend(self.connection_tree[current])
        return visited


    def _get_edges_within(
        self,
        nodes: Set[str]
    ) -> List[TreeEdge]:
        """
        Returns a list of all edges where both nodes are within a given node set.

        Args:
            nodes (Set[str]): Set of node names.

        Returns:
            List[TreeEdge]: List of edges (as frozensets) internal to the subtree.
        """
        return [edge for edge in self.edges if edge.issubset(nodes)]


    def _write_edges_to_json(
        self,
        edges: List[TreeEdge],
        outfile: str
    ) -> None:
        """
        Writes a list of edges and their distribution info to a JSON file.

        Args:
            edges (List[TreeEdge[str]): List of edges to serialize.
            outfile (str): Path to the output JSON file.
        """
        serialized_data = {}

        for edge in edges:
            u, v = sorted(edge)
            edge_key = f"{u},{v}"
            distribution_type = self.edges[edge].impl.type
            parameters = self.edges[edge].params

            serialized_data[edge_key] = {
                "distribution": distribution_type,
                "parameters": parameters
            }
        with open(outfile, 'w', encoding='utf-8') as f:
            json.dump(serialized_data, f, indent=4)

    def get_equivalent_class(
        self,
        first_obs: str,
        outfile: str
    ) -> List[str]:
        """
        Computes the equivalence class of nodes reachable from the first infected observer,
        filters all edges within that subtree, and writes them to a JSON file.

        Args:
            first_obs (str): The first observer infected.
            outfile (str): File path where the resulting subtree edges will be written.

        Returns:
            List[str]: List of observer nodes within the equivalence class.
        """
        nodes = self._get_subtree_nodes(first_obs)
        edges = self._get_edges_within(nodes)
        self._write_edges_to_json(edges, outfile)
        return list(nodes & set(self.observers))

    def save(
        self,
        outfile: str
    ) -> None:
        """
        Saves the entire tree structure to a JSON file in the same format used
        by get_equivalent_class(), using all edges currently in the tree.

        Args:
            outfile (str): Path to the output JSON file.
        """
        all_edges = list(self.edges.keys())
        self._write_edges_to_json(all_edges, outfile)


    def objective_function(
        self,
        u: ArrayLike,
        source: str,
        method: str = None
    ) -> float:
        """
        Objective function used to identify the most likely infection source.

        Args:
            u (ArrayLike): Vector to evaluate the objective function at.
            source (str): Candidate infection source node.
            method (Optional[str]): Augmentation method to apply (default is None):
                None: No augmentation,
                'linear': Linear approximation,
                'exponential': Exponential approximation,
                'exact': Exact solution for iid exponential delays.

        Returns:
            float: Value of the objective function at `u`.
        """
        val0 = self.joint_mgf(u, source)
        t = list(self.infection_times.values())
        val1 = np.exp(-1*np.dot(u,t))
        if method is not None:
            val1 = val1*((len(self.observers)-1)/(2*len(self.observers)-1))
            conditional_expectation = 0
            for o in self.observers:
                conditional_expectation += self.cond_joint_mgf(u,source,o,method = method)
            conditional_expectation = conditional_expectation*(1/(2*len(self.observers)-1))
            val1 += conditional_expectation
        val = -1*(val1-val0)**2
        return val

    def localize(
        self,
        method: str = None
    ) -> str:
        """
        Estimates the most likely infection source node by minimizing the objective function.

        Args:
            method (Optional[str]): Augmentation method to use (default is None):
                None: No augmentation,
                'linear': Linear approximation,
                'exponential': Exponential approximation,
                'exact': Exact solution for iid exponential delays.

        Returns:
            str: Name of the predicted source node.
        """
        results = {}
        for node in self.nodes:
            res = sp.optimize.minimize(
                fun=self.objective_function,
                x0=np.random.rand(len(self.observers)),
                args=(node, method),
                bounds=[(0, None)] * len(self.observers),
                method='Nelder-Mead'
            )
            results[node] = res.fun

        return max(results, key=results.get)
