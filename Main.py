import networkx as nx
import GenerateGraph
import Constants
import Agent

def mapGraphToAgents():
    agents = Agent.generateAgents()
    DiGraph = GenerateGraph.generateSBMGraph()
    for u, v in DiGraph.edges():
        agents[u].addFollower(v)
    return agents

def simulationProper(agents):
    for currentTime in range(Constants.MAXIMUM_TIME):
        for agentID in range(Constants.N_AGENTS):
            currentAgent = agents[agentID]

            if (currentAgent.getCurrentStatus(currentTime) == "Online"):
                currentAgent.processFeed(currentTime)

if __name__ == "__main__":
    agentsList = mapGraphToAgents()
    simulationProper(agentsList)