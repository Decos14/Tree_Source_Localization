import numpy as np
import scipy as sp
import csv
import math
from typing import Dict, List, Tuple, Callable, Union, FrozenSet
from numpy.typing import NDArray, ArrayLike
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
        self.A = self.build_A_matrix()

    def build_tree(
        self, 
        file_name: str
    ) -> TreeDatastructure:
        """
        A function to build the tree data structure from a file

        Args:
            file_name (str): A string containing the path to the file to build the tree from.

        Returns:
            TreeDatastructure: The tree datastructure built from the file.
        """
        with open(file_name, newline='') as filestream:
            reader = csv.reader(filestream)

            parameters = {}
            distributions = {}
            edge_delays = {}
            edge_mgfs = {}
            edge_mgf_derivatives = {}
            edge_mgf_derivatives_2 = {}
            edges = []
            nodes = set()

            for curr in reader:
                if not curr:
                    continue
                
                edge = frozenset({curr[0],curr[1]})
                edges.append(edge)
                nodes = nodes.union(edge)
                edge_delays[edge] = 0
                dist = curr[2]
                distributions[edge] = dist
                if dist == 'N':
                    mu = float(curr[3])
                    sigma2 = float(curr[4])
                    parameters[edge] = {
                        'mu': float(curr[3]),
                        'sigma2': float(curr[4])
                    }
                    edge_mgfs[edge] = lambda t, mu=mu, sigma2=sigma2: 1 if np.isclose(t, 0) else MGF_Functions.PositiveNormalMGF(t, mu, sigma2)
                    edge_mgf_derivatives[edge] =  lambda t, mu=mu, sigma2=sigma2: 1 if np.isclose(t, 0) else MGF_Functions.PositiveNormalMGFDerivative(t, mu, sigma2)
                    edge_mgf_derivatives_2[edge] = lambda t, mu=mu, sigma2=sigma2: 1 if np.isclose(t, 0) else MGF_Functions.PositiveNormalMGFDerivative2(t,mu,sigma2)

                if dist == 'E':
                    lam = float(curr[3])
                    parameters[edge] = {
                        'lambda': float(curr[3])
                    }
                    edge_mgfs[edge] = lambda t, lam=lam: MGF_Functions.ExponentialMGF(t, lam)
                    edge_mgf_derivatives[edge] =lambda t, lam=lam: MGF_Functions.ExponentialMGFDerivative(t, lam)
                    edge_mgf_derivatives_2[edge] = lambda t, lam=lam: MGF_Functions.ExponentialMGFDerivative2(t, lam)

                if dist == 'U':
                    a = float(curr[3])
                    b = float(curr[4])
                    parameters[edge] = {
                        'start': float(curr[3]),
                        'stop': float(curr[4])
                    }
                    edge_mgfs[edge] = lambda t, start=a, stop=b: 1 if np.isclose(t, 0) else MGF_Functions.UniformMGF(t, start, stop)
                    edge_mgf_derivatives[edge] = lambda t, start=a, stop=b: 1 if np.isclose(t, 0) else MGF_Functions.UniformMGFDerivative(t, start, stop)
                    edge_mgf_derivatives_2[edge] = lambda t, start=a, stop=b: 1 if np.isclose(t, 0) else MGF_Functions.UniformMGFDerivative2(t, start, stop)

                if dist == 'P':
                    lam = float(curr[3])
                    parameters[edge] = {
                        'lambda': float(curr[3])
                    }
                    edge_mgfs[edge] = lambda t, lam=lam: MGF_Functions.PoissonMGF(t, lam)
                    edge_mgf_derivatives[edge] = lambda t, lam=lam: MGF_Functions.PoissonMGFDerivative(t, lam)
                    edge_mgf_derivatives_2[edge] = lambda t, lam=lam: MGF_Functions.PoissonMGFDerivative2(t, lam)
                    
                if dist == 'C':
                    sigma2 = float(curr[3])
                    parameters[edge] = {
                        'sigma2': float(curr[3])
                    }
                    edge_mgfs[edge] = lambda t, sigma2=sigma2: 1 if np.isclose(t, 0) else MGF_Functions.AbsoluteCauchyMGF(t,sigma2)
                    edge_mgf_derivatives[edge] = lambda t, sigma2=sigma2: 1 if np.isclose(t, 0) else MGF_Functions.AbsoluteCauchyMGFDerivative(t,sigma2)
                    edge_mgf_derivatives_2[edge] = lambda t, sigma2=sigma2: 1 if np.isclose(t, 0) else MGF_Functions.AbsoluteCauchyMGFDerivative2(t,sigma2)

            self.edges = edges
            self.nodes = list(nodes)
            self.distributions = distributions
            self.parameters = parameters
            self.edge_delays = edge_delays
            self.edge_mgfs = edge_mgfs
            self.edge_mgf_derivatives = edge_mgf_derivatives
            self.edge_mgf_derivatives_2 = edge_mgf_derivatives_2
        
    def build_connection_tree(self) -> None:
        """
        Takes in a tree datastructure and returns the adjacency dictionary of the topology of the tree

        Args:
            tree (TreeDatastructure): The tree datastructure to compute the adjacency dictionary of.

        Returns:
            None: Sets self.connection tree
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
    
    def build_A_matrix(self) -> NDArray[np.integer]:
        """
        Constructs the A tensor of the tree with a specific observer set

        Returns:
            NDArray[np.int]: A matrix with 1 in an entry if the edge is on the path between the corresponding observer and potential source
        """
        A = {}
        for _, node in enumerate(self.nodes):
            A_layer= np.zeros((len(self.observers),len(self.edges)))
            for j, obs in enumerate(self.observers):
                for k, edge in enumerate(self.edges):
                    path = self.path_edge(self.DFS(node,obs))
                    if edge in path:
                        A_layer[j,k]=1  
            A[node] = A_layer 
        return A
    
    def simulate_edge(
        self,
        edge: TreeEdge
    ) -> Union[float,int]:
        """
        Simulate the edge delay of a specific edge

        Args:
            edge (TreeEdge): The edge to simulate
        Returns:
            float: The edge delay of the given edge
        """
        distribution = self.distributions[edge]
        parameters = self.parameters[edge]
        if distribution == 'N':
            return np.random.normal(parameters['mu'],parameters['sigma2'])
        
        if distribution == 'E':
            return np.random.exponential(parameters['lambda'])
        
        if distribution == 'U':
            return np.random.uniform(parameters['start'],parameters['stop'])
        
        if distribution == 'P':
            return np.random.poisson(parameters['lambda'])
        
        if distribution == 'C':
            return np.abs(sp.stats.cauchy.rvs(loc=0,scale = parameters['sigma2']))

    def simulate(self) -> None:
        """
        Simulate all of the edges of the tree and update the tree
        """
        for edge in self.edges:
            self.edge_delays[edge]=self.simulate_edge(edge)

    def DFS(
        self,
        source: str,
        observer: str
    ) -> List[str]:
        """
        Runs a DFS search on the tree to find the path between an observer and a potential source node

        Args:
            source (str): The potential source node.
            observer (str): The observer.

        Returns:
            List[str]: A list containing the sequence of nodes that form the path from the source to the observer.
        """
        stack = [(source, [source])]
        visited = set()
        while stack:
            (vertex, path) = stack.pop()
            if vertex not in visited:
                if vertex == observer:
                    return path
                visited.add(vertex)
                for neighbor in self.connection_tree[vertex]:
                    stack.append((neighbor,path + [neighbor]))
    
    def path_edge(self,
        path: List[str]
    ) -> List[TreeEdge]:
        """
        A function that converts a path from a sequence of nodes to a sequence of edges

        Args:
            path (List[str]): A list of nodes forming a path through the tree
        
        Returns:
            List[TreeEdge]: A list of edges forming a path through the tree
        """
        edges = []
        for i in range(len(path)-1):
            edges.append(frozenset({path[i],path[i+1]}))
        return edges
    
    #Simulates the infection from a given source node to an observer node
    def Infection_Simulation(
        self,
        source: str
    ) -> None:
        """
        Simulates the infection from a given source node to the observers

        Args:
            source (str): Name of the source node to simulate the infection from
        
        Returns:
            None: Updates the value in self.infection_times
        """
        infection_times = {}

        #Finds the path each observer to the source using DFS and then adds up the edge costs stored in the tree on those paths
        for observer in self.observers:
            path = self.DFS(source, observer)
            edges = self.path_edge(path)
            time = 0
            for edge in edges:
                time += self.edge_delays[edge]
                #print(time)
            infection_times[observer] = time
        self.infection_times= infection_times

    def joint_mgf(
        self,
        u: ArrayLike,
        source: str
    ) -> float:
        """
        Computes the Joint Moment Generating Function of the infection times of the observers
        from a given source at a given value.

        Args:
            u (ArrayLike): The vector to evaluate the Joint MGF at
            source: The potential source of the infection

        Returns:
            float: The value of the Joint MGF at u.
        """
        mgf = 1
        for i,edge in enumerate(self.edges):
            tempval = np.matmul(u,self.A[source][:,i])
            if tempval != 0:
                mgf *= self.edge_mgfs[edge](tempval)
        return mgf
    
    def cond_joint_mgf(
        self,
        u: ArrayLike,
        source: str,
        obs_o: str,
        method: int
    ) -> float:
        """
        Compute (or approximate) the conditional Joint Moment Generating function given the first
        infected observer, of the observers from the given source at a given value

        Args:
            u (Arraylike[float]): The value to evaluate the conditional Joint MGF at.
            source (str): The potential source to assume the infection began at.
            obs_o: The name of the first infected observer.
            method: The augmentation method to choose
                    (1: Linear approximation,
                     2: Exponential approximation,
                     3: Exact solution for iid exponential delays)

        Returns:
            float: the conditional Joint MGF evaluated at u.
        """
        mgf = 1
        val1 = -1
        for i,node in enumerate(self.nodes):
            if node == source:
                val1 = i

        path  = self.path_edge(self.DFS(source, obs_o))
        for i,edge in enumerate(self.edges):
            if edge not in path:
                tempval = np.matmul(u,self.A[source][:,i])
                if tempval != 0:
                    mgf *= self.edge_mgfs[edge](tempval)
        
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
                b2+= self.edge_mgf_derivatives_2[edge](0)-self.edge_mgf_derivatives[edge](0)**2
                b1+= np.matmul(u,self.A[source][:,i])*b2
            b = b1/b2
            a1 = 0
            for i,edge in enumerate(self.edges):
                a1+=(b-np.matmul(u,self.A[source][:,i]))*self.edge_mgf_derivatives[edge](0)
            a = np.exp(a1)
            mgf *= a*np.exp(-1*b*self.infection_times[obs_o])

        if method == 3 and len(path) != 0:
            Theta = np.zeros((len(path),len(path)))
            lam = -1
            prod = 1
            for i, edge in enumerate(path):
                if self.distributions[edge] != 'E':
                    raise ValueError(f"Non exponential distribution: {self.distributions[edge]}. Distribution must be exponential")
                if i == 0:
                    lam = self.parameters[edge]['lambda']
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
        Computes the Equivalence class of the tree that is sufficient for estimating the source
        and where the true source is guaranteed to exist.

        Args:
            first_obs (str): The name of the first infected observer.
            outfile (str): The path of the file to write the equivalent class to.
        
        Returns:
            List[str]: A list of the observers that are still relevant.
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
        with open(outfile, 'w') as file:
            for edge in include_edge:
                file.write(f"{list(edge)[0]},{list(edge)[1]},{self.distributions[edge]},{','.join(map(str,self.parameters[edge].values()))}\n")
        return list(nodes.intersection(set(self.observers)))
        
    
    def obj_func(
        self,
        u: ArrayLike,
        source: str,
        augment: int = None
    ) -> float:
        """
        The objective function to minimize to find the true source

        Args:
            u (ArrayLike): The vector to evaluate the objective function at.
            source (str): The potential source to assume is the origin of the infection.
            augment (int): What augmentation to use, default is None:
                    (None: None,
                     1: Linear approximation,
                     2: Exponential approximation,
                     3: Exact solution for iid exponential delays)

        Returns:
            float: The value of the objective function at u
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
        Localize the true source of an infection

        Args:
            method: The augmentation method, defaults to None.
                    (None: None,
                     1: Linear approximation,
                     2: Exponential approximation,
                     3: Exact solution for iid exponential delays)

        Returns:
            str: The name of the predicted true source node.
        """
        m = np.zeros(len(self.nodes))
        for i, node in enumerate(self.nodes):
            m[i] = sp.optimize.minimize(self.obj_func, np.random.rand(len(self.observers)), args = (node,method),bounds = [(0,None) for i in range(len(self.observers))],method='Nelder-Mead').fun
        return self.nodes[np.argmax(m)]