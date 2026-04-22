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


def buildSimulationData(postsQueue, configData):
    """
    Initialises the simulationData structure for all posts.

    Per-post (posts):
      interactions_over_time       — interaction count at each absolute simulation
                                     tick (length = MAXIMUM_TIME).
      interactions_by_belief_camp  — how many interacting agents came from each
                                     belief camp.
      total_interactions           — filled in after the simulation ends.

    contested_agents:
      Per-postID counts of agents that received ALL posts and chose to share
      only that post. Also rolls up into misinfo_only / realnews_only / both /
      neither for the campaign-level comparison.

    misinfo_vs_realnews:
      Aggregated view of the three-wave misinfo campaign vs. the single real
      news post. Built entirely in post-simulation aggregation so the hot-path
      in acceptPost() stays unchanged.
    """
    # Identify which postIDs belong to misinfo and which to real news.
    # These sets are used both here and in collectContestedAgents().
    misinfo_ids   = set()
    realnews_ids  = set()
    for post in configData["Posts"]:
        if post["misinformation"]:
            misinfo_ids.add(post["postID"])
        else:
            realnews_ids.add(post["postID"])

    simulationData = {
        # ── per-post granular data ────────────────────────────────────────────
        "posts": {
            p.postID: {
                "interactions_over_time":   [0] * Constants.MAXIMUM_TIME,
                "interactions_by_belief_camp": {
                    "red": 0, "centrist": 0, "blue": 0
                },
                "total_interactions": 0,
            }
            for p in postsQueue
        },

        # ── contested agents ─────────────────────────────────────────────────
        # Per-postID: agents that received ALL posts and shared only that post.
        # Campaign-level rollups are added after the per-postID counts are done.
        "contested_agents": {
            **{f"chose_post_{p.postID}": 0 for p in postsQueue},
            "chose_misinfo_only":  0,   # received all, shared ≥1 misinfo, 0 real news
            "chose_realnews_only": 0,   # received all, shared ≥1 real news, 0 misinfo
            "chose_both":          0,   # received all, shared from both sides
            "chose_neither":       0,   # received all, shared nothing
        },

        # ── campaign-level comparison (filled post-simulation) ────────────────
        "misinfo_vs_realnews": {
            "total_misinfo_interactions":   0,
            "total_realnews_interactions":  0,
            "misinfo_interactions_by_camp": {"red": 0, "centrist": 0, "blue": 0},
            "realnews_interactions_by_camp":{"red": 0, "centrist": 0, "blue": 0},
            "misinfo_combined_over_time":   [0] * Constants.MAXIMUM_TIME,
            "realnews_over_time":           [0] * Constants.MAXIMUM_TIME,
        },

        # ── bookkeeping (not written to output, used internally) ─────────────
        "_misinfo_ids":  list(misinfo_ids),
        "_realnews_ids": list(realnews_ids),
    }

    return simulationData


def collectContestedAgents(simulationData, agentsList, allPostIDs):
    """
    Walk every agent after the simulation ends and classify those who
    received ALL posts into one of the contest outcomes.

    Per-postID counts (chose_post_N):
      Agent received all posts and shared ONLY that single post.

    Campaign-level rollups:
      chose_misinfo_only  — shared at least one misinfo post, zero real news
      chose_realnews_only — shared at least one real news post, zero misinfo
      chose_both          — shared from both sides
      chose_neither       — shared nothing from the contested set
    """
    allPostIDs    = set(allPostIDs)
    misinfo_ids   = set(simulationData["_misinfo_ids"])
    realnews_ids  = set(simulationData["_realnews_ids"])

    for agent in agentsList:
        if agent.isNewsAgency:
            continue

        # Only count agents exposed to every post in the run
        if not allPostIDs.issubset(agent.receivedPosts):
            continue

        shared_misinfo   = misinfo_ids  & agent.sharedPosts
        shared_realnews  = realnews_ids & agent.sharedPosts
        shared_all_posts = allPostIDs   & agent.sharedPosts

        # ── per-postID: shared exactly this one post and nothing else ─────────
        for postID in allPostIDs:
            if agent.sharedPosts & allPostIDs == {postID}:
                simulationData["contested_agents"][f"chose_post_{postID}"] += 1

        # ── campaign-level rollup ─────────────────────────────────────────────
        has_misinfo  = len(shared_misinfo)  > 0
        has_realnews = len(shared_realnews) > 0

        if has_misinfo and has_realnews:
            simulationData["contested_agents"]["chose_both"] += 1
        elif has_misinfo:
            simulationData["contested_agents"]["chose_misinfo_only"] += 1
        elif has_realnews:
            simulationData["contested_agents"]["chose_realnews_only"] += 1
        else:
            simulationData["contested_agents"]["chose_neither"] += 1


def aggregateMisinfoVsRealNews(simulationData):
    """
    Fills misinfo_vs_realnews by summing across the per-post data that
    acceptPost() already populated during the simulation.

    Kept separate from the simulation loop so the hot-path stays clean.
    """
    misinfo_ids  = set(simulationData["_misinfo_ids"])
    realnews_ids = set(simulationData["_realnews_ids"])
    mvr          = simulationData["misinfo_vs_realnews"]

    for postID, postData in simulationData["posts"].items():
        if postID in misinfo_ids:
            mvr["total_misinfo_interactions"] += postData["total_interactions"]
            for camp in ("red", "centrist", "blue"):
                mvr["misinfo_interactions_by_camp"][camp] += (
                    postData["interactions_by_belief_camp"][camp]
                )
            for t, count in enumerate(postData["interactions_over_time"]):
                mvr["misinfo_combined_over_time"][t] += count

        elif postID in realnews_ids:
            mvr["total_realnews_interactions"] += postData["total_interactions"]
            for camp in ("red", "centrist", "blue"):
                mvr["realnews_interactions_by_camp"][camp] += (
                    postData["interactions_by_belief_camp"][camp]
                )
            for t, count in enumerate(postData["interactions_over_time"]):
                mvr["realnews_over_time"][t] += count


def simulationProper(configData, networkData, simulationAgentsList: "list[Agent.Agent]", agenciesList: "list[Agent.Agent]"):
    
    postsQueue = readPosts(configData, simulationAgentsList, agenciesList)
    simulationData = buildSimulationData(postsQueue, configData)
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

    # ── post-simulation aggregation ───────────────────────────────────────────
    for postID, postData in simulationData["posts"].items():
        postData["total_interactions"] = sum(postData["interactions_over_time"])

    collectContestedAgents(simulationData, simulationAgentsList, allPostIDs)
    aggregateMisinfoVsRealNews(simulationData)

    # ── strip internal bookkeeping keys before writing output ─────────────────
    simulationData.pop("_misinfo_ids",  None)
    simulationData.pop("_realnews_ids", None)

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