import networkx as nx
import matplotlib.pyplot as plt
import Constants

def plotGraph(G, title):
    """Visualize a directed graph."""
    plt.figure(figsize=(6, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=80, node_color='skyblue', edgecolors='black')
    nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=10, alpha=0.6)
    plt.title(title)
    plt.axis('off')
    plt.show()

def removeSelfLoops(G):
    """Remove (u,u) edges if any exist."""
    G.remove_edges_from(nx.selfloop_edges(G))
    return G

# --- 1. Directed Barabási–Albert Model ---
def generateBAGraph(n=100, m=2, seed=None):
    """
    n = total nodes
    m = edges each new node forms
    """
    G = nx.barabasi_albert_graph(n, m, seed=seed)
    D = nx.DiGraph((u, v) for u, v in G.edges() if u != v)
    removeSelfLoops(D)
    # plotGraph(D, f"Directed Barabási–Albert Model (n={n}, m={m}, seed={seed})")
    return D

# --- 2. Directed Holme–Kim Model ---
def generateHKGraph(n=100, m=3, p=0.3, seed=None):
    """
    n = total nodes
    m = edges per new node
    p = prob. of forming triangles (clustering)
    """
    G = nx.powerlaw_cluster_graph(n, m, p, seed=seed)
    D = nx.DiGraph((u, v) for u, v in G.edges() if u != v)
    removeSelfLoops(D)
    # plotGraph(D, f"Directed Holme–Kim Model (n={n}, m={m}, p={p}, seed={seed})")
    return D

# --- 3. Directed Stochastic Block Model ---
def generateSBMGraph(
    sizes=[30, 30, 40],
    probs=[[0.8, 0.05, 0.02],
           [0.05, 0.7, 0.03],
           [0.02, 0.03, 0.9]],
    seed=None
):
    """
    sizes = nodes per group [[s1], [s2], ...]
    probs = edge probability matrix
    """

    G = nx.stochastic_block_model(sizes, probs, directed=True, seed=seed)
    removeSelfLoops(G)
    # plotGraph(G, f"Directed Stochastic Block Model (sizes={sizes}, seed={seed})")
    return G

if __name__ == "__main__":
    ba_digraph = generateBAGraph(
        Constants.N_AGENTS, 
        Constants.BA_NEW_EDGES, 
        Constants.GRAPH_SEED
    )
    hk_digraph = generateHKGraph(
        Constants.N_AGENTS, 
        Constants.HK_NEW_EDGES, 
        Constants.HK_PROB_CLUSTERING, 
        Constants.GRAPH_SEED
    )
    sbm_digraph = generateSBMGraph(
        Constants.SBM_SIZES,
        Constants.SBM_PROB_MATRIX,
        Constants.GRAPH_SEED
    )
