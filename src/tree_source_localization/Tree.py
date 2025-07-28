import numpy as np
import scipy as sp
import csv
import math
from typing import Dict, List, Tuple, Callable, Union, FrozenSet
from numpy.typing import ArrayLike
from .Search import _DepthFirstSearch
from .EdgeDistribution import EdgeDistribution
from tree_source_localization import MGF_Functions


# Structure is [(distribution, parameters), [mgf, mgf', mgf''], edge delay]
TreeValue = List[Union[Union[Tuple[str, float], Tuple[str, float, float]], Tuple[Callable[[float], float], Callable[[float], float], Callable[[float],float]], float]]
#Structure is {node, node}
TreeEdge = FrozenSet[str]

TreeDatastructure = Dict[TreeEdge, TreeValue]

class Tree:
    def __init__(self,file_name, observers, infection_times):
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
    ) -> TreeDatastructure:
        """
        Builds the tree data structure from a CSV file, initializing edges, nodes,
        distributions, parameters, delays, and moment generating functions as instance variables.

        Args:
            file_name (str): Path to the CSV file describing the tree.
        """
        with open(file_name, newline='', encoding='utf-8') as filestream:
            reader = csv.reader(filestream)

            self.edges = []
            self.nodes = set()
            self.edge_distributions = {}

            for row in reader:
                if not row or row[0].startswith('#'):
                    continue

                node1, node2 = row[0], row[1]
                dist_type = row[2]
                raw_params = row[3]

                # Convert "mu=3.0;sigma2=1.0" → {'mu': 3.0, 'sigma2': 1.0}
                param_dict = {
                    k: float(v)
                    for k, v in (item.split('=') for item in raw_params.split(';'))
                }

                edge = frozenset({node1, node2})
                self.edges.append(edge)
                self.nodes.update(edge)
                self.edge_distributions[edge] = EdgeDistribution(dist_type, param_dict)

            self.nodes = list(self.nodes)
        
    def build_connection_tree(self) -> None:
        """
        Builds an adjacency dictionary representing the tree topology and assigns it
        to the instance variable `self.connection_tree`.
        """
        connection_tree = {}
        for edge in self.edges:
            nodes = []
            for node in edge:
                nodes.append(node)
            
            if nodes[0] not in connection_tree:
                connection_tree[nodes[0]] = [nodes[1]]
            if nodes[1] not in connection_tree:
                connection_tree[nodes[1]] = [nodes[0]]
            if nodes[0] in connection_tree and nodes[1] not in connection_tree[nodes[0]]:
                connection_tree[nodes[0]].append(nodes[1])
            if nodes[1] in connection_tree and nodes[0] not in connection_tree[nodes[1]]:
                connection_tree[nodes[1]].append(nodes[0])
        self.connection_tree = connection_tree
    
    def build_A_matrix(self) -> None:
        """
        Constructs the A-matrix tensor of the tree using the observers and assigns it
        to the instance variable `self.A`.
        """
        A_matrix = {}
        for _, node in enumerate(self.nodes):
            A_layer= np.zeros((len(self.observers),len(self.edges)))
            for j, obs in enumerate(self.observers):
                for k, edge in enumerate(self.edges):
                    path = self.search.get_path(node,obs)
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
            self.edge_distributions[edge].sample()
    
    #Simulates the infection from a given source node to an observer node
    def Infection_Simulation(
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

        #Finds the path each observer to the source using self.search and then adds up the edge costs stored in the tree on those paths
        for observer in self.observers:
            edges = self.search.get_path(source, observer)
            time = 0
            for edge in edges:
                time += self.edge_distributions[edge].delay
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
        for i,edge in enumerate(self.edges):
            tempval = np.matmul(u,self.A[source][:,i])
            if tempval != 0:
                mgf *= self.edge_distributions[edge].mgf(tempval)
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
        for i,edge in enumerate(self.edges):
            if edge not in path:
                tempval = np.matmul(u,self.A[source][:,i])
                if tempval != 0:
                    mgf *= self.edge_distributions[edge].mgf(tempval)
        
        if method == 1 and len(path) != 0:
            tempval = 0
            for i,edge in enumerate(self.edges):
                tempval += np.matmul(u,self.A[source][:,i])
            tempval *= -self.infection_times[obs_o]/(len(path))
            mgf *= np.exp(tempval)

        if method == 2 and len(path) != 0:
            b1 = 0
            b2=0
            for i,edge in enumerate(self.edges):
                if self.edge_distributions[edge].impl.type == "C":
                    raise ValueError(f"Cannot use method 2 with the AbsoluteCauchy distribution")
                b2+= self.edge_distributions[edge].mgf_derivative2(0)-self.edge_distributions[edge].mgf_derivative(0)**2
                b1+= np.matmul(u,self.A[source][:,i])*b2
            b = b1/b2
            a1 = 0
            for i,edge in enumerate(self.edges):
                a1+=(b-np.matmul(u,self.A[source][:,i]))*self.edge_distributions[edge].mgf_derivative(0)
            a = np.exp(a1)
            mgf *= a*np.exp(-1*b*self.infection_times[obs_o])

        if method == 3 and len(path) != 0:
            Theta = np.zeros((len(path),len(path)))
            lam = -1
            prod = 1
            for i, edge in enumerate(path):
                if self.edge_distributions[edge].impl.type != "E":
                    raise ValueError(f"Non exponential distribution: {self.edge_distributions[edge].impl.type}. Distribution must be exponential")
                if i == 0:
                    lam = self.edge_distributions[edge].params['lambda']
                prod *= 1/(lam + np.matmul(u,self.A[source][:,i]))
                Theta[i,i] = -1*(lam + np.matmul(u,self.A[source][:,i]))
                if i != len(path)-1:
                    Theta[i,i+1] = lam + np.matmul(u,self.A[source][:,i])
            alpha = np.zeros((1,len(path)))
            alpha[0,0]=1
            t = self.infection_times[obs_o]
            exp_Theta = sp.linalg.expm(t*Theta)
            g_t = -1*np.matmul(np.matmul(np.matmul(alpha, exp_Theta),Theta),np.ones((1,len(path),1)))
            mgf *= g_t*(t**(len(path)-1))*np.exp(-1*lam*t)*(math.factorial(len(path)-1))*prod
        return mgf
    
    def Equivalent_Class(
        self,
        first_obs: str,
        outfile: str
    ) -> List[str]:
        """
        Computes the equivalence class of nodes sufficient for source estimation, based
        on the first infected observer, and writes the relevant subtree edges to a file.

        Args:
            first_obs (str): The first observer infected.
            outfile (str): File path where the equivalence class subtree will be written.

        Returns:
            List[str]: List of relevant observers within the equivalence class.
        """
        to_check = self.connection_tree[first_obs]
        nodes = set(first_obs)
        while to_check:
            if to_check[0] in nodes:
                to_check.pop(0)
            elif to_check[0] in self.observers:
                nodes.add(to_check[0])
                to_check.pop(0)
            else:
                nodes.add(to_check[0])
                to_check.extend(self.connection_tree[to_check[0]])
                to_check.pop(0)
        include_edge = []
        for edge in self.edges:
            edge_nodes = list(edge)
            if edge_nodes[0] in nodes and edge_nodes[1] in nodes:
                include_edge.append(edge)
        with open(outfile, 'w', encoding='utf-8') as file:
            for edge in include_edge:
                file.write(f"{list(edge)[0]},{list(edge)[1]},{self.edge_distributions[edge].impl.type},{','.join(map(str,self.edge_distributions[edge].params.values()))}\n")
        return list(nodes.intersection(set(self.observers)))


    def obj_func(
        self,
        u: ArrayLike,
        source: str,
        augment: int = None
    ) -> float:
        """
        Objective function used to identify the most likely infection source.

        Args:
            u (ArrayLike): Vector to evaluate the objective function at.
            source (str): Candidate infection source node.
            augment (int, optional): Augmentation method to apply (default is None):
                None: No augmentation,
                1: Linear approximation,
                2: Exponential approximation,
                3: Exact solution for iid exponential delays.

        Returns:
            float: Value of the objective function at `u`.
        """
        val0 = self.joint_mgf(u, source)
        t = list(self.infection_times.values())
        val1 = np.exp(-1*np.dot(u,t))
        if augment != None:
            val1 = val1*((len(self.observers)-1)/(2*len(self.observers)-1))
            tempval = 0
            for o in self.observers:
                tempval += self.cond_joint_mgf(u,source,o,method = augment)
            tempval = tempval*(1/(2*len(self.observers)-1))
            val1 += tempval
        val = -1*(val1-val0)**2
        return val

    def localize(
        self,
        method = None
    ) -> str:
        """
        Estimates the most likely infection source node by minimizing the objective function.

        Args:
            method (int, optional): Augmentation method to use (default is None):
                None: No augmentation,
                1: Linear approximation,
                2: Exponential approximation,
                3: Exact solution for iid exponential delays.

        Returns:
            str: Name of the predicted source node.
"""
        m = np.zeros(len(self.nodes))
        for i, node in enumerate(self.nodes):
            m[i] = sp.optimize.minimize(self.obj_func, np.random.rand(len(self.observers)), args = (node,method),bounds = [(0,None) for i in range(len(self.observers))],method='Nelder-Mead').fun
        return self.nodes[np.argmax(m)]