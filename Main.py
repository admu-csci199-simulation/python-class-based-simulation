import random
import Agent
import Constants
import GenerateGraph
import Post
import Statistics

def mapGraphToAgents():
    agents = Agent.generateAgents()
    DiGraph = GenerateGraph.generateSBMGraph()
    for u, v in DiGraph.edges():
        agents[u].addFollower(agents[v])
    return agents

def setupPosts(agentsList):
    for i in range(Constants.N_INITIAL_POSTS):
        chosenPosterID = random.randint(0, Constants.N_AGENTS-1)
        print(chosenPosterID)
        currentPost = Post.generatePost(
            originalPoster = chosenPosterID,
            beliefValue = agentsList[chosenPosterID].beliefValue
        )
        print(agentsList[chosenPosterID].beliefValue)
        agentsList[chosenPosterID].addPostToFeedBuffer(currentPost)

def simulationProper(simulationAgentsList):
    for agentID in range(Constants.N_AGENTS):
        simulationAgentsList[agentID].addNewPostsToFeedQueue()

    for currentTime in range(Constants.MAXIMUM_TIME):
        for agentID in range(Constants.N_AGENTS):
            currentAgent = simulationAgentsList[agentID]
            
            if (currentAgent.isOnline(currentTime)):
                currentAgent.processFeed(currentTime)
        
        for agentID in range(Constants.N_AGENTS):
            currentAgent = simulationAgentsList[agentID]
            currentAgent.addNewPostsToFeedQueue()

if __name__ == "__main__":
    agentsList = mapGraphToAgents()
    stats = Statistics.Statistics(agentsList)
    stats.generateBeliefTypePieChart(saveDir="old")
    
    setupPosts(agentsList)
    simulationProper(agentsList)

    stats.generatePerTickGraphs()
    stats.generateBinnedGraphs()
    stats.generateAgentDemogGraph()
    stats.generateBeliefTypePieChart(saveDir="new")