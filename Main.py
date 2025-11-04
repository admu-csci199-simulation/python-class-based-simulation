import random
import Agent
import Constants
import GenerateGraph
import Post
import Statistics
from collections import deque

def mapGraphToAgents():
    agents = Agent.generateAgents()
    DiGraph = GenerateGraph.generateSBMGraph()
    for u, v in DiGraph.edges():
        agents[u].addFollower(agents[v])
    return agents

def setupPosts(agentsList):
    postsList = []
    for i in range(Constants.N_INITIAL_POSTS):
        chosenPosterID = random.randint(0, Constants.N_AGENTS-1) # Is it possible to skew this so that chosenPoster is more likely to be an active agent
        # print(chosenPosterID)
        postsList.append(
            Post.generatePost(
                postID = i,
                originalPoster = chosenPosterID,
                beliefValue = agentsList[chosenPosterID].beliefValue
            )
        )
    
    postsList.sort(key=lambda post: post.postingTime)
    postsQueue = deque(postsList)
    return postsQueue

def simulationProper(postsQueue, simulationAgentsList):
    for currentTime in range(Constants.MAXIMUM_TIME):
        while (postsQueue[0].postingTime == currentTime):
            currentPost = postsQueue[0]
            simulationAgentsList[currentPost.originalPoster].sharePost(currentPost)
            postsQueue.popleft()

        for agentID in range(Constants.N_AGENTS):
            currentAgent = simulationAgentsList[agentID]
            
            if (currentAgent.isOnline(currentTime)):
                currentAgent.processFeed(currentTime)
        
        for agentID in range(Constants.N_AGENTS):
            currentAgent = simulationAgentsList[agentID]
            currentAgent.addNewPostsToFeedQueue()

if __name__ == "__main__":
    agentsList = mapGraphToAgents()
    postsQueue = setupPosts(agentsList)
    
    stats = Statistics.Statistics(agentsList)
    stats.generateBeliefTypePieChart(saveDir="old")
    
    simulationProper(postsQueue, agentsList)

    stats.generatePerTickGraphs()
    stats.generateBinnedGraphs()
    stats.generateAgentDemogGraph()
    stats.generateBeliefTypePieChart(saveDir="new")