import networkx as nx
from Agent import Agent
import Constants


class SimGraph:
    def __init__(self, agents, nx_digraph: "nx.DiGraph"):
        """
        agents: list of Agent instances.
        nx_digraph: NetworkX DiGraph.
        G = simGraph(agents, generateGraph.generateBAGraph(...))
        """
        # Assume len(agents) == len(nx_digraph.nodes()) == Constants.N_AGENTS
        self.agents = agents
        self.n = Constants.N_AGENTS

        # Initialize adjacency list for each agent
        self.adjList = [[] for _ in range(self.n)]

        # Copy structure from nx_digraph
        # Assume nodes in nx_digraph are 0..N-1 which matches agents' indices
        for u, v in nx_digraph.edges():
            self.adjList[u].append(v)
