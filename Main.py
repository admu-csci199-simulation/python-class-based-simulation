import os
import argparse
import json
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

def randomizePosts(agentsList):
    postsList = []

    for i in range(Constants.N_INITIAL_POSTS):
        chosenPosterID = random.randint(0, Constants.N_AGENTS-1) # Is it possible to skew this so that chosenPoster is more likely to be an active agent
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

def simulationProper(postsQueue, simulationAgentsList : list[Agent.Agent]):
    for currentTime in range(Constants.MAXIMUM_TIME):
        # OPs posts their original posts at time currentTime
        while (len(postsQueue) > 0 and postsQueue[0].postingTime == currentTime):
            currentPost = postsQueue[0]
            nthLayer = 0 # all posts here are original posts, thus layer is 0
            simulationAgentsList[currentPost.originalPoster].sharePost((currentPost, nthLayer))
            simulationAgentsList[currentPost.originalPoster].sharedPosts.add(currentPost.postID)
            postsQueue.popleft()
            #continue

        # all agents process what is in their feed at time currentTime
        for agentID in range(Constants.N_AGENTS):
            currentAgent = simulationAgentsList[agentID]
            
            if (currentAgent.isOnline(currentTime)):
                currentAgent.processFeed(currentTime)
        
        # transfer all new posts from feedBuffer to feedQueue for all agents 
        for agentID in range(Constants.N_AGENTS):
            currentAgent = simulationAgentsList[agentID]
            currentAgent.addNewPostsToFeedQueue()
        
        if (currentTime <= 50):
            saveDir = os.path.join(Constants.GRAPHS_DIR, 'animation')
            filename = str(currentTime).zfill(4)
            stats.generateBeliefTypePieChart(saveDir=Constants.GIF_FRAMES_DIR, filename=filename, addLabels=False)

def setupParameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("--custom_post")
    args = parser.parse_args()
    
    parameters = {}
    if args.custom_post:        
        parameters["custom_post"] = args.custom_post

    return parameters


def getCustomPosts(agentsList, filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    
    postsQueue = deque([])
    for post in data["Posts"]:
        postsQueue.append(
            Post.generatePost(
                postID = post["postID"],
                postingTime= post["postingTime"],
                originalPoster = random.randint(0, Constants.N_AGENTS-1),
                beliefValue = post["beliefValue"],
                interestValue = post["interestValue"],
                postTopic = post["postTopic"]
            )
        )
    return postsQueue


if __name__ == "__main__":
    parameters = setupParameters()

    agentsList = mapGraphToAgents()
    stats = Statistics.Statistics(agentsList)

    if "custom_post" in parameters:
        postsQueue = getCustomPosts(agentsList, parameters["custom_post"])
    else:
        postsQueue = randomizePosts(agentsList)
    
    stats.generateGraphs(saveDir=Constants.PRE_SIM_GRAPHS_DIR)
    simulationProper(postsQueue, agentsList)
    stats.generateGraphs(saveDir=Constants.POST_SIM_GRAPHS_DIR)