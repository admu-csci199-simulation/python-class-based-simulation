from Graph import SimGraph
import GenerateGraph
import Constants


def mapGraphToAgents():
    agents = []
    simulationGraph = SimGraph(agents, GenerateGraph.generateSBMGraph())
    mappedAgents = []
    return mappedAgents

def simulationProper(agents):
    for currentTime in range(Constants.MAXIMUM_TIME):
        for agentID in range(Constants.N_AGENTS):
            currentAgent = agents[agentID]

            if (currentAgent.getCurrentStatus(currentTime) == "Online"):
                currentAgent.processFeed(currentTime)

if __name__ == "__main__":
    agentsList = mapGraphToAgents()
    simulationProper(agentsList)