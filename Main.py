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


def buildSimulationData(postsQueue):
    """
    Initialises the simulationData structure for all posts.

    Per-post:
      interactions_over_time    — interaction count at each absolute simulation
                                  tick (length = MAXIMUM_TIME). Using absolute
                                  time means both posts share the same x-axis,
                                  making direct comparison straightforward.
      interactions_by_belief_camp — how many interacting agents came from each
                                  camp; reveals cross-camp reach.
      total_interactions        — filled in after the simulation ends.

    contested_agents — agents that received BOTH posts and had to choose.
      Computed post-simulation from agent.receivedPosts vs agent.sharedPosts.
      This is the cleanest "who won the fight" signal because it only counts
      agents who genuinely faced a choice.
    """
    simulationData = {
        "posts": {
            p.postID: {
                "interactions_over_time": [0] * Constants.MAXIMUM_TIME,
                "interactions_by_belief_camp": {"red": 0, "centrist": 0, "blue": 0},
                "total_interactions": 0,
            }
            for p in postsQueue
        },
        "contested_agents": {
            "chose_post_0":  0,  # received both, shared only post 0
            "chose_post_1":  0,  # received both, shared only post 1
            "chose_both":    0,  # received both, shared both
            "chose_neither": 0,  # received both, shared neither
        },
    }
    return simulationData


def collectContestedAgents(simulationData, agentsList, allPostIDs):
    """
    Walk every agent after the simulation ends and classify those who
    received all competing posts into one of the four contest outcomes.
    Only agents who were exposed to every post in the run are counted —
    agents who only saw one post had no real choice to make.
    """
    allPostIDs = set(allPostIDs)

    for agent in agentsList:
        if agent.isNewsAgency:
            continue

        # Skip agents that were not exposed to all posts
        if not allPostIDs.issubset(agent.receivedPosts):
            continue

        sharedAll    = allPostIDs.issubset(agent.sharedPosts)
        sharedNone   = agent.sharedPosts.isdisjoint(allPostIDs)
        sharedPost0  = 0 in agent.sharedPosts
        sharedPost1  = 1 in agent.sharedPosts

        if sharedAll:
            simulationData["contested_agents"]["chose_both"] += 1
        elif sharedNone:
            simulationData["contested_agents"]["chose_neither"] += 1
        elif sharedPost0:
            simulationData["contested_agents"]["chose_post_0"] += 1
        elif sharedPost1:
            simulationData["contested_agents"]["chose_post_1"] += 1


def simulationProper(configData, networkData, simulationAgentsList: "list[Agent.Agent]", agenciesList: "list[Agent.Agent]"):
    
    postsQueue = readPosts(configData, simulationAgentsList, agenciesList)
    simulationData = buildSimulationData(postsQueue)
    allPostIDs = [p.postID for p in postsQueue]

    for currentTime in range(Constants.MAXIMUM_TIME):
        # OPs post their original posts at time currentTime
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

    # ── post-simulation aggregation ──────────────────────────────────────
    for postID, postData in simulationData["posts"].items():
        postData["total_interactions"] = sum(postData["interactions_over_time"])

    collectContestedAgents(simulationData, simulationAgentsList, allPostIDs)

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

if __name__ == "__main__":
    conf_name = sys.argv[1]
    output_name = sys.argv[2]

    runSimulation(conf_name, output_name)