import os, json, sys
from Random import rng as random
import Agent, Post, Constants, GenerateGraph, Statistics
import copy
from collections import deque
from math import ceil

def mapGraphToNewsAgencies(agentsList, connectionsCount):
    numAgencies = ceil(len(agentsList)/100)
    newsAgencies = Agent.generateNewsAgencies(numAgencies)

    if connectionsCount > len(agentsList):
        raise RuntimeError("News Agencies connection count is greater than the number of agents.")
    
    for agencyIdx in range(numAgencies):
        for agentIdx in random.sample(range(len(agentsList)), connectionsCount):
            newsAgencies[agencyIdx].addFollower(agentsList[agentIdx])

    return newsAgencies

def mapGraphToAgents(agentsData, networkData):
    agents = Agent.generateAgents(agentsData)
    for agentIdx in range(agentsData["Agent Count"]):
        networkData["agents"].append(
            {
                "response_type" : agents[agentIdx].classifyAgentType()
            }
        )

    DiGraph = GenerateGraph.generateSBMGraph(
            sizes = [agentsData["Red Count"], agentsData["Centrist Count"], agentsData["Blue Count"]]
        )
    for u, v in DiGraph.edges():
        agents[u].addFollower(agents[v])
        networkData["network_edges"].append([u, v])
    return agents


def simulationProper(configData, networkData, simulationAgentsList: "list[Agent.Agent]", agenciesList: "list[Agent.Agent]"):
    
    postsQueue = readPosts(configData, simulationAgentsList, agenciesList)
    simulationData = {p.postID : [0 for i in range(24*60)] for p in postsQueue} # for hypothesis 1

    
    for currentTime in range(Constants.MAXIMUM_TIME):
        # OPs posts their original posts at time currentTime
        while (len(postsQueue) > 0 and postsQueue[0].postingTime == currentTime):
            currentPost = postsQueue[0]

            # Assign post OP at time {currentTime}
            if currentPost.isMisinformation:
                currentPost.assignMisinfoOP(simulationAgentsList)
            else:
                currentPost.assignRealNewsOP(agenciesList)


            # OP shares to its neighbors
            nthLayer = 0 # all posts here are original posts, thus layer is 0
            if currentPost.isMisinformation:
                simulationAgentsList[currentPost.originalPoster].sharePost(currentPost, nthLayer, currentTime, networkData)
                simulationAgentsList[currentPost.originalPoster].sharedPosts.add(currentPost.postID)
            else:
                agenciesList[currentPost.originalPoster].sharePost(currentPost, nthLayer, currentTime, networkData)
                agenciesList[currentPost.originalPoster].sharedPosts.add(currentPost.postID)
            postsQueue.popleft()

        # all agents process what is in their feed at time currentTime
        for agentID in range(configData["Agents"]["Agent Count"]):
            currentAgent = simulationAgentsList[agentID]
            
            if (currentAgent.isOnline(currentTime)):
                currentAgent.processFeed(networkData, currentTime, simulationData)
        
        # transfer all new posts from feedBuffer to feedQueue for all agents 
        for agentID in range(configData["Agents"]["Agent Count"]):
            currentAgent = simulationAgentsList[agentID]
            currentAgent.addNewPostsToFeedQueue()


    return simulationData


def readPosts(configData, agentsList, agenciesList):
    postsList = []

    for post in configData["Posts"]:
        postsList.append(
            Post.generatePost(
                postID=post["postID"],
                postingTime=post["postingTime"],
                beliefValue=post["beliefValue"],
                interestValue=post["interestValue"],
                postTopic=post["postTopic"],
                isMisinformation=post["misinformation"]
            )
        )

    postsList.sort(key=lambda post: post.postingTime)
    postsQueue = deque(postsList)

    return postsQueue

def runSimulation(conf_name, output_name):

    with open(os.path.join("input", conf_name), "r") as f:
        configData = json.load(f)
    
    # data for network visualziation
    networkData = {
        "agents" : [],
        "network_edges" : [],
        "agent_states" : {},
        "share_events" : {}
    }

    agentsList = mapGraphToAgents(configData["Agents"], networkData)
    agenciesList = mapGraphToNewsAgencies(agentsList, int(len(agentsList) * Constants.NEWS_AGENCY_PERCENTAGE))

    simulationData = simulationProper(configData, networkData, agentsList, agenciesList)

    with open(os.path.join("output", output_name), "w") as f:
        json.dump(simulationData, f, indent=4)
    # print(f"Simulation data saved in output/{output_name}.json")

    # with open("output/network_output.json", "w") as f:
    #     json.dump(networkData, f, indent=4)
    # print("Network data saved in output/network_output.json")

if __name__ == "__main__":
    # try:
    conf_name = sys.argv[1]
    output_name = sys.argv[2]

    runSimulation(conf_name, output_name)
    # except:
    #     print("Error on command line arguments.")