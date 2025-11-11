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
    # Access block assignments from the graph
    # print(DiGraph.nodes)
    # block_assignments = [DiGraph.nodes[node].get('block') for node in DiGraph.nodes()]
    # for v in block_assignments:
    #     print(v)
    for u, v in DiGraph.edges():
        agents[u].addFollower(agents[v])
    return agents

def setupPosts(agentsList):
    postsList = []
    
    # for choosing a certain belief value
    chosenBeliefValues = [-4]
    for i in range(len(chosenBeliefValues)):
        for agentID in range(Constants.N_AGENTS):
            if agentsList[agentID].beliefValue == chosenBeliefValues[i]:
                postsList.append(
                    Post.generatePost(
                        postID = i,
                        postingTime = 0,
                        originalPoster =  agentID,
                        beliefValue = chosenBeliefValues[i]
                    )
                )
                break

    # for i in range(Constants.N_INITIAL_POSTS):
    #     chosenPosterID = random.randint(0, Constants.N_AGENTS-1) # Is it possible to skew this so that chosenPoster is more likely to be an active agent
    #     postsList.append(
    #         Post.generatePost(
    #             postID = i,
    #             postingTime=0, # just for this one case
    #             originalPoster = chosenPosterID,
    #             beliefValue = agentsList[chosenPosterID].beliefValue

    #         )
    #     )
    
    postsList.sort(key=lambda post: post.postingTime)
    postsQueue = deque(postsList)
    print("Posts:", postsList)
    return postsQueue

def simulationProper(postsQueue, simulationAgentsList):
    for currentTime in range(Constants.MAXIMUM_TIME):
        while (len(postsQueue) > 0 and postsQueue[0].postingTime == currentTime):
            currentPost = postsQueue[0]
            simulationAgentsList[currentPost.originalPoster].sharePost(currentPost)
            simulationAgentsList[currentPost.originalPoster].sharedPosts.add(currentPost.postID)
            postsQueue.popleft()
            #continue

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
    postsQueue = setupPosts(agentsList)
    
    stats.generateGraphs(saveDir=Constants.PRE_SIM_GRAPHS_DIR)
    simulationProper(postsQueue, agentsList)
    stats.generateGraphs(saveDir=Constants.POST_SIM_GRAPHS_DIR)