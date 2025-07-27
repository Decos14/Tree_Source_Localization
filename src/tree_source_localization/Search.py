from typing import List, FrozenSet

TreeEdge = FrozenSet[str]

class _DepthFirstSearch():
    def __init__(self, connection_tree):
        self.cached_paths = {}
        self.connection_tree = connection_tree

    def get_path(
        self,
        start: str,
        stop: str
        ):
        '''
        Uses a lazy cache to find and store if new the path between two nodes

        Args:
            start (str): The start of the path
            stop (str): The end of the path
        
        Returns:
            List[TreeEdge]: Sequence of edges representing the path between start and stop.
        '''
        key = (start, stop)

        if key not in self.cached_paths:
            path = self._dfs(start, stop)
            self.cached_paths[key] = path
            self.cached_paths[(stop, start)] = list(reversed(path))
        
        return self.cached_paths[key]
    
    def _path_edge(self,
        path: List[str]
    ) -> List[TreeEdge]:
        """
        Converts a path given as a sequence of nodes into a sequence of edges.

        Args:
            path (List[str]): Sequence of nodes forming a path.

        Returns:
            List[TreeEdge]: Sequence of edges representing the path.
        """
        return [frozenset((path[i], path[i+1])) for i in range(len(path) - 1)]

    def _dfs(
        self,
        source: str,
        observer: str
    ) -> List[str]:
        """
        Finds the path from a source node to an observer using depth-first search.

        Args:
            source (str): The starting node.
            observer (str): The destination (observer) node.

        Returns:
            List[TreeEdge]: Sequence of edges representing the path.
        """
        stack = [(source, [source])]
        visited = set()
        while stack:
            (vertex, path) = stack.pop()
            if vertex not in visited:
                if vertex == observer:
                    return self._path_edge(path)
                visited.add(vertex)
                for neighbor in self.connection_tree[vertex]:
                    stack.append((neighbor,path + [neighbor]))