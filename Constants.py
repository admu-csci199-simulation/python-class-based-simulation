from math import lcm
# from Random import rng as random
import time, os

# Flag to check if simulation should:
# TRUE  -> update agent belief value before sharing
# FALSE -> update agent belief value after sharing
UPDATE_BELIEF_VALUE_BEFORE_SHARE = True

# NETWORK STRUCTURE CONSTANTS
GRAPH_SEED = 100
# random.seed(GRAPH_SEED)

# divisible by:
#   3 -     # of blocks/clusters
#   100 -   # 90-9-1 rule and agentType ratios
N_AGENTS = lcm(3, 100)

BA_NEW_EDGES = 2
HK_NEW_EDGES = 2
HK_PROB_CLUSTERING = 0.3
SBM_SIZES = [ N_AGENTS//3, N_AGENTS//3, N_AGENTS//3 ]
SBM_PROB_MATRIX = [
    [0.18, 0.03, 0.03],
    [0.03, 0.18, 0.03],
    [0.03, 0.03, 0.18],
]

# AGENT GENERATION PARAMETERS
# assumed 3 clusters
BELIEF_VALUES = [(-4, -2), (-1, 1), (2, 4)]


AGENT_TYPE = {
    "gullible" : {
        "steepnessRange" : (1, 1),
        "toleranceRange" : (8, 9),
        "count" : int(N_AGENTS*0.15)
    },
    "normal" : {
        "steepnessRange" : (2, 2),
        "toleranceRange" : (4, 7),
        "count" : int(N_AGENTS*0.65)
    },
    "stubborn" : {
        "steepnessRange" : (3, 4),
        "toleranceRange" : (1, 3),
        "count" : int(N_AGENTS*0.20)
    }
}

AGENT_SHARE_PROPENSITY = {
    "lurker" : {
        "propensityRange" : (0, 24),
        "count" : int(N_AGENTS*0.90)
    },
    "normal" : {
        "propensityRange" : (25, 54),
        "count" : int(N_AGENTS*0.09)
    },
    "active" : {
        "propensityRange" : (55, 100),
        "count" : int(N_AGENTS*0.01)
    }
}

AGENT_DURATION = {
    "online" : (1, 8),
    "offline" : (2, 16)
}

AGENT_BELIEF_TYPE = {
    "red": [-4, -3, -2],
    "centrist": [-1, 0, 1],
    "blue": [2, 3, 4]
}

# POSTS PARAMETERS
N_INITIAL_POSTS = 1
N_TOPICS = 5

# SIMULATION PARAMETERS
MAXIMUM_TIME = 48*60 # 48 hours in mins


# STATISTICAL GRAPHS SETTINGS
BIN_SIZE = 15
FIG_SIZE = (8, 5)
GRAPHS_DIR = os.path.join("output", f'graphs_{time.strftime('%Y-%m-%d_%H-%M-%S')}')
PRE_SIM_GRAPHS_DIR = os.path.join(GRAPHS_DIR, "pre_sim")
POST_SIM_GRAPHS_DIR = os.path.join(GRAPHS_DIR, "post_sim")

# GIF
GIF_FRAMES_DIR = f"{GRAPHS_DIR}/frames"
GIF_OUT_PATH = f"{GRAPHS_DIR}/belief_type_pie_chart.gif" 
GIF_DURATION = 2

# NEWS AGENCIES
NEWS_AGENCY_PERCENTAGE = 0.75 # percentage over agents count