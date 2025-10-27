from Agent import Agent

class SimGraph:
    def __init__(self, nx_digraph: "nx.DiGraph"):
        """
        Initialize a custom graph using a NetworkX DiGraph.
        """
        self.adjList = {node: [] for node in self.nodes}

        # Copy edges to adjacency list
        for u, v in nx_digraph.edges():
            self.adj_list[u].append(v)
