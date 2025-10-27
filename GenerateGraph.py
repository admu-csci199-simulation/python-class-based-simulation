import networkx as nx
import matplotlib.pyplot as plt
import Constants

def plotGraph(G, title):
    """Visualize a directed graph."""
    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=80, node_color='skyblue', edgecolors='black')
    nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=10, alpha=0.6)
    plt.title(title)
    plt.axis('off')
    plt.show()

def plotSBMGraph(G, title):
    """Visualize a directed graph with node labels and block-based colors."""
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, seed=42)

    # Each node's block is stored as an attribute by the SBM generator
    blocks = [G.nodes[n]['block'] for n in G.nodes()]

    # Draw nodes, coloring by block
    nx.draw_networkx_nodes(
        G, pos,
        node_size=400,
        node_color=blocks,     # color by block index
        cmap=plt.cm.tab10,     # choose a categorical color map
        edgecolors='black'
    )

    # Draw edges
    nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=10, alpha=0.6)

    # Add node labels (node indices)
    labels = {node: str(node) for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

    plt.title(title)
    plt.axis('off')
    plt.show()

def removeSelfLoops(G):
    """Remove (u,u) edges if any exist."""
    G.remove_edges_from(nx.selfloop_edges(G))
    return G

# --- 1. Directed Barabási–Albert Model ---
def generateBAGraph(n=Constants.N_AGENTS, m=Constants.BA_NEW_EDGES, seed=Constants.GRAPH_SEED):
    """
    n = total nodes
    m = edges each new node forms
    """
    G = nx.barabasi_albert_graph(n, m, seed=seed)
    D = nx.DiGraph((u, v) for u, v in G.edges() if u != v)
    removeSelfLoops(D)
    plotGraph(D, f"Directed Barabási–Albert Model (n={n}, m={m}, seed={seed})")
    return D

# --- 2. Directed Holme–Kim Model ---
def generateHKGraph(n=Constants.N_AGENTS, m=Constants.HK_NEW_EDGES, p=Constants.HK_PROB_CLUSTERING, seed=Constants.GRAPH_SEED, plot=False):
    """
    n = total nodes
    m = edges per new node
    p = prob. of forming triangles (clustering)
    """
    G = nx.powerlaw_cluster_graph(n, m, p, seed=seed)
    D = nx.DiGraph((u, v) for u, v in G.edges() if u != v)
    removeSelfLoops(D)
    plotGraph(D, f"Directed Holme–Kim Model (n={n}, m={m}, p={p}, seed={seed})")
    return D

# --- 3. Directed Stochastic Block Model ---
def generateSBMGraph( sizes=Constants.SBM_SIZES, probs=Constants.SBM_PROB_MATRIX, seed=Constants.GRAPH_SEED, plot=False):
    """
    sizes = nodes per group [[s1], [s2], ...]
    probs = edge probability matrix
    """

    G = nx.stochastic_block_model(sizes, probs, directed=True, seed=seed)
    removeSelfLoops(G)
    if plot: plotSBMGraph(G, f"Directed Stochastic Block Model (sizes={sizes}, seed={seed})")
    return G

if __name__ == "__main__":
    # ba_digraph = generateBAGraph()
    # hk_digraph = generateHKGraph()
    sbm_digraph = generateSBMGraph(
        sizes=[5, 5, 5],
        probs=[
            [0.5, 0.05, 0.05],
            [0.05, 0.4, 0.05],
            [0.05, 0.05, 0.6],
        ],
        seed=12831254,
        plot=True
    )
