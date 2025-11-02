import networkx as nx
import GenerateGraph
import GeneratePosts
import Constants
import Agent
import random

def mapGraphToAgents():
    agents = Agent.generateAgents()
    DiGraph = GenerateGraph.generateSBMGraph()
    for u, v in DiGraph.edges():
        agents[u].addFollower(agents[v])
    return agents

def setupPosts(agentsList):
    for i in range(Constants.N_INITIAL_POSTS):
        chosenPosterID = random.randint(0, Constants.N_AGENTS-1)
        currentPost = GeneratePosts.generatePost(
            originalPoster = chosenPosterID,
            beliefValue = agentsList[chosenPosterID].beliefValue
        )
        agentsList[chosenPosterID].addPostToFeed(currentPost, -1)

def simulationProper(simulationAgentsList):
    for currentTime in range(Constants.MAXIMUM_TIME):
        for agentID in range(Constants.N_AGENTS):
            currentAgent = simulationAgentsList[agentID]
            currentAgent.addNewPostsToFeed(currentTime)

            if (currentAgent.isOnline(currentTime)):
                currentAgent.processFeed(currentTime)

if __name__ == "__main__":
    agentsList = mapGraphToAgents()
    setupPosts(agentsList)
    simulationProper(agentsList)