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


def generateStaticData(configData):
    return {
        "post_count": len(configData["Posts"]),
        "agent_count": configData["Agents"]["Agent Count"],
        "agent_response_type": {
            agentType: count
            for agentType, count in [("gullible", configData["Agents"]["Gullible Count"]), ("normal", configData["Agents"]["Normal Count"]), ("stubborn", configData["Agents"]["Stubborn Count"])]
        },
        "minutes": Constants.MAXIMUM_TIME
    }


def generateStaticPostInformation(postsQueue):
    staticPostInformation = {}
    for post in postsQueue:
        postID = post.postID

        staticPostInformation[f'post_{postID}'] = {
            "isMisinformation": post.isMisinformation,
            "post_camp": post.classifyBeliefCamp(),
            "original_post_time" : post.postingTime,
        }

    return staticPostInformation

# this function is obsolete. If going to use, code according to configdata
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
    
    postsList.sort(key=lambda post: (post.postingTime, post.postID))
    postsQueue = deque(postsList)
    return postsQueue


def simulationProper(configData, networkData, simulationAgentsList: "list[Agent.Agent]", agenciesList: "list[Agent.Agent]"):
    postsQueue = readPosts(configData, simulationAgentsList, agenciesList)

    simulationData = {
        "static_data": {},
        "static_post_information": {},
        "post_seen_data": {},
        "post_shared_data": {},
        "dynamic_data": []
    }
    simulationData["static_data"] = generateStaticData(configData)
    simulationData["static_post_information"] = generateStaticPostInformation(postsQueue)

    # Dictionary for layer_information
    cumulativePostLayerCounts = {
        post["postID"]: {}
        for post in configData["Posts"]
    }

    # Dictionary for post_type_count_per_camp
    cumulativePostTypeCountPerCamp = {
        "red": {"misinformation": 0, "regular": 0, "interactions": 0},
        "centrist": {"misinformation": 0, "regular": 0, "interactions": 0},
        "blue": {"misinformation": 0, "regular": 0, "interactions": 0}
    }

    # Track last processed interaction index per agent
    lastProcessedInteractionIndex = {
        agentID: 0
        for agentID in range(configData["Agents"]["Agent Count"])
    }

    # Dictionary for post_seen_data and post_shared_data
    campIndex = {
        "red": 0,
        "centrist": 1,
        "blue": 2
    }
    postSeenData = {}
    postSharedData = {}

    for post in configData["Posts"]:
        postID = post["postID"]

        postSeenData[f"post_{postID}"] = []
        postSharedData[f"post_{postID}"] = []

    postSeenCounters = {
        post["postID"]: [0,0,0]
        for post in configData["Posts"]
    }

    postSharedCounters = {
        post["postID"]: [0,0,0]
        for post in configData["Posts"]
    }

    # Start of simulation
    for currentTime in range(Constants.MAXIMUM_TIME):
        # OPs posts their original posts at time currentTime
        while (len(postsQueue) > 0 and postsQueue[0].postingTime == currentTime):
            currentPost = postsQueue[0]

            # Assign post OP at time {currentTime}
            if currentPost.isMisinformation:
                currentPost.assignMisinfoOP(simulationAgentsList)
            else:
                currentPost.assignRealNewsOP(agenciesList)

            posterID = currentPost.originalPoster
            posterCamp = currentPost.classifyBeliefCamp()

            # Update published regular/misinformation post count of camps
            if currentPost.isMisinformation:
                cumulativePostTypeCountPerCamp[posterCamp]["misinformation"] += 1
            else:
                # always a news agency. Belief value of a news agency is always 0, so posterCamp will always be centrist
                cumulativePostTypeCountPerCamp[posterCamp]["regular"] += 1

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
                currentAgent.processFeed(networkData, currentTime)
        
        # transfer all new posts from feedBuffer to feedQueue for all agents 
        for agentID in range(configData["Agents"]["Agent Count"]):
            currentAgent = simulationAgentsList[agentID]

            # Update post_seen_data
            for post, layer in currentAgent.feedBuffer:
                postID = post.postID
                camp = currentAgent.classifyAgentBelief()
                idx = campIndex[camp]

                postSeenCounters[postID][idx] += 1

            currentAgent.addNewPostsToFeedQueue()

        for agentID in range(configData["Agents"]["Agent Count"]):
            agent = simulationAgentsList[agentID]
            startIdx = lastProcessedInteractionIndex[agentID]
            newInteractions = agent.interactionsDone[startIdx:]

            for interaction in newInteractions:
                postID = interaction.post.postID
                layer = interaction.layer
                posterID = interaction.post.originalPoster
                postCamp = interaction.post.classifyBeliefCamp()
                posterCamp = simulationAgentsList[posterID].classifyAgentBelief()
                isMisinfo = interaction.post.isMisinformation

                # Update layer_information count
                if interaction.isShared:
                    if layer not in cumulativePostLayerCounts[postID]:
                        cumulativePostLayerCounts[postID][layer] = 0
                    cumulativePostLayerCounts[postID][layer] += 1

                # Updated post_shared_data
                if interaction.isShared:
                    postID = interaction.post.postID
                    camp = agent.classifyAgentBelief()
                    idx = campIndex[camp]

                    postSharedCounters[postID][idx] += 1

                cumulativePostTypeCountPerCamp[postCamp]["interactions"] += 1

            lastProcessedInteractionIndex[agentID] = len(agent.interactionsDone)


        # Dictionary for camp_distribution
        campDistribution = {
            "red": {"gullible": 0, "normal": 0, "stubborn": 0},
            "centrist": {"gullible": 0, "normal": 0, "stubborn": 0},
            "blue": {"gullible": 0, "normal": 0, "stubborn": 0}
        }

        for agent in simulationAgentsList:
            camp = agent.classifyAgentBelief()
            agentType = agent.classifyAgentType()
            campDistribution[camp][agentType] += 1

        # Dictionary for layer_information
        postInformation = {}
        for post in configData["Posts"]:
            postID = post["postID"]

            # Sort layers ascending
            layers = sorted(cumulativePostLayerCounts[postID].keys())
            layerInfo = [cumulativePostLayerCounts[postID][layer] for layer in layers]

            postInformation[f"post_{postID}"] = {
                "layer_information": layerInfo
            }
                        
        snapshot = {
            "camp_distribution": campDistribution,
            "post_type_count_per_camp": copy.deepcopy(cumulativePostTypeCountPerCamp),
            "post_information": postInformation
        }

        simulationData["dynamic_data"].append(snapshot)

        # Update post_seen_data and post_shared_data
        for post in configData["Posts"]:
            postID = post["postID"]

            postSeenData[f"post_{postID}"].append(
                postSeenCounters[postID].copy()
            )

            postSharedData[f"post_{postID}"].append(
                postSharedCounters[postID].copy()
            )

        simulationData["post_seen_data"] = postSeenData
        simulationData["post_shared_data"] = postSharedData

        # for networkData agent_states
        for agentID in range(configData["Agents"]["Agent Count"]):
            agent = simulationAgentsList[agentID]
            if str(agentID) not in networkData["agent_states"]:
                networkData["agent_states"][str(agentID)] = []

            networkData["agent_states"][str(agentID)].append(agent.classifyAgentBelief())

        # if (currentTime <= 50):
        #     saveDir = os.path.join(Constants.GRAPHS_DIR, 'animation')
        #     filename = str(currentTime).zfill(4)
        #     stats.generateBeliefTypePieChart(saveDir=Constants.GIF_FRAMES_DIR, filename=filename, addLabels=False)

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
    try:
        conf_name = sys.argv[1]
        output_name = sys.argv[2]
        runSimulation(conf_name, output_name)
    except:
        print("Error on command line arguments.")