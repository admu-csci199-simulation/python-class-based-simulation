import os
import json
import random
import Agent
import Constants
import GenerateGraph
import Post
import Statistics
import copy
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


def simulationProper(postsQueue, simulationAgentsList : list[Agent.Agent], stats):
    simulationData = {
        "static_data": {},
        "static_post_information": {},
        "dynamic_data": []
    }

    with open(os.path.join("input", "config.json"), "r") as f:
        configData = json.load(f)

    simulationData["static_data"] = generateStaticData(configData)
    simulationData["static_post_information"] = generateStaticPostInformation(postsQueue)

    # Track cumulative layer counts per post
    cumulativePostLayerCounts = {
        post["postID"]: {}
        for post in configData["Posts"]
    }

    # Track cumulative post type counts per camp
    cumulativePostTypeCountPerCamp = {
        "red": {"misinformation": 0, "regular": 0, "interactions": 0},
        "centrist": {"misinformation": 0, "regular": 0, "interactions": 0},
        "blue": {"misinformation": 0, "regular": 0, "interactions": 0}
    }

    # Track last processed interaction index per agent
    lastProcessedInteractionIndex = {
        agentID: 0
        for agentID in range(Constants.N_AGENTS)
    }

    # Start of simulation
    for currentTime in range(Constants.MAXIMUM_TIME):
        # OPs posts their original posts at time currentTime
        while (len(postsQueue) > 0 and postsQueue[0].postingTime == currentTime):
            currentPost = postsQueue[0]

            posterID = currentPost.originalPoster
            posterCamp = currentPost.classifyBeliefCamp()

            if currentPost.isMisinformation:
                cumulativePostTypeCountPerCamp[posterCamp]["misinformation"] += 1
            else:
                cumulativePostTypeCountPerCamp[posterCamp]["regular"] += 1

            nthLayer = 0 # all posts here are original posts, thus layer is 0
            simulationAgentsList[currentPost.originalPoster].sharePost(currentPost, nthLayer)
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

        for agentID in range(Constants.N_AGENTS):
            agent = simulationAgentsList[agentID]
            startIdx = lastProcessedInteractionIndex[agentID]
            newInteractions = agent.interactionsDone[startIdx:]

            for interaction in newInteractions:
                postID = interaction.post.postID
                layer = interaction.layer
                posterID = interaction.post.originalPoster
                posterCamp = simulationAgentsList[posterID].classifyAgentBelief()
                isMisinfo = interaction.post.isMisinformation

                # Update cumulative layer count
                if layer not in cumulativePostLayerCounts[postID]:
                    cumulativePostLayerCounts[postID][layer] = 0
                cumulativePostLayerCounts[postID][layer] += 1

                cumulativePostTypeCountPerCamp[posterCamp]["interactions"] += 1

            lastProcessedInteractionIndex[agentID] = len(agent.interactionsDone)

        campDistribution = {
            "red": {"gullible": 0, "normal": 0, "stubborn": 0},
            "centrist": {"gullible": 0, "normal": 0, "stubborn": 0},
            "blue": {"gullible": 0, "normal": 0, "stubborn": 0}
        }

        for agent in simulationAgentsList:
            camp = agent.classifyAgentBelief()
            agentType = agent.classifyAgentType()
            campDistribution[camp][agentType] += 1

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


        # if (currentTime <= 50):
        #     saveDir = os.path.join(Constants.GRAPHS_DIR, 'animation')
        #     filename = str(currentTime).zfill(4)
        #     stats.generateBeliefTypePieChart(saveDir=Constants.GIF_FRAMES_DIR, filename=filename, addLabels=False)

    return simulationData

def generateStaticData(configData):
    return {
        "post_count": len(configData["Posts"]),
        "agent_count": Constants.N_AGENTS,
        "agent_response_type": {
            agentType: values["count"]
            for agentType, values in Constants.AGENT_TYPE.items()
        },
        "minutes": Constants.MAXIMUM_TIME
    }

def generateStaticPostInformation(postsQueue):
    staticPostInformation = {}
    for post in postsQueue:
        postID = post.postID

        staticPostInformation[f'post_{postID}'] = {
            "isMisinformation": post.isMisinformation,
            "post_camp": post.classifyBeliefCamp()
        }

    return staticPostInformation


def readPosts(agentsList):
    filename = os.path.join("input", "config.json")
    with open(filename, 'r') as f:
        data = json.load(f)

    postsList = []

    for post in data["Posts"]:
        postsList.append(
            Post.generatePost(
                postID=post["postID"],
                postingTime=post["postingTime"],
                originalPoster=random.randint(0, Constants.N_AGENTS-1),
                beliefValue=post["beliefValue"],
                interestValue=post["interestValue"],
                postTopic=post["postTopic"],
                isMisinformation=post["misinformation"]
            )
        )

    postsList.sort(key=lambda post: post.postingTime)
    postsQueue = deque(postsList)

    return postsQueue


def runSimulation():
    agentsList = mapGraphToAgents()
    postsQueue = readPosts(agentsList)

    stats = Statistics.Statistics(agentsList)
    
    #stats.generateGraphs(saveDir=Constants.PRE_SIM_GRAPHS_DIR)
    simulationData = simulationProper(postsQueue, agentsList, stats)
    #stats.generateGraphs(saveDir=Constants.POST_SIM_GRAPHS_DIR)


    with open("output/simulation_output.json", "w") as f:
        json.dump(simulationData, f, indent=4)
    print("Simulation data saved in output/simulation_output.json")


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