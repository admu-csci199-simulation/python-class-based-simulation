from math import e as EULER 
from math import exp, pow
from Post import Post
from Helper import BernoulliTrial
from PostInteraction import PostInteraction
import Constants
from Random import rng as random

def generateNewsAgencies(size):
    agencies = [Agent(id=i, isNewsAgency=True) for i in range(size)]
    
    # news agencies are active 24/7
    for ai in range(size):
        agencies[ai].setActiveDuration(Constants.MAXIMUM_TIME, 0, "Online", 0)
    
    return agencies


def generateAgents(agentsData):
    agents = [Agent(i) for i in range(agentsData["Agent Count"])]

    # Set belief values
    for agentIdx in range(agentsData["Red Count"]):
        minBeliefValue, maxBeliefValue = Constants.BELIEF_VALUES[0]
        assignedBeliefValue = random.randint(minBeliefValue, maxBeliefValue)
        agents[agentIdx].setBeliefValue(assignedBeliefValue)
    for agentIdx in range(agentsData["Red Count"], agentsData["Red Count"] + agentsData["Centrist Count"]):
        minBeliefValue, maxBeliefValue = Constants.BELIEF_VALUES[1]
        assignedBeliefValue = random.randint(minBeliefValue, maxBeliefValue)
        agents[agentIdx].setBeliefValue(assignedBeliefValue)
    for agentIdx in range(agentsData["Red Count"] + agentsData["Centrist Count"], agentsData["Red Count"] + agentsData["Centrist Count"] + agentsData["Blue Count"]):
        minBeliefValue, maxBeliefValue = Constants.BELIEF_VALUES[2]
        assignedBeliefValue = random.randint(minBeliefValue, maxBeliefValue)
        agents[agentIdx].setBeliefValue(assignedBeliefValue)
    
    assert(agentsData["Agent Count"] == agentsData["Red Count"] + agentsData["Centrist Count"] + agentsData["Blue Count"])
    
    # Set steepness tolerance
    idxRandom = [i for i in range(agentsData["Agent Count"])] # randomized agents index 
    random.shuffle(idxRandom)
    idx = 0
    for _ in range(agentsData["Gullible Count"]):
        minSteepness, maxSteepness = Constants.AGENT_TYPE["gullible"]["steepnessRange"]
        minTolerance, maxTolerance = Constants.AGENT_TYPE["gullible"]["toleranceRange"]
        assignedSteepness = random.randint(minSteepness, maxSteepness)
        assignedTolerance = random.randint(minTolerance, maxTolerance)
        agents[idxRandom[idx]].setSteepnessTolerance(assignedSteepness, assignedTolerance)
        idx += 1
    for _ in range(agentsData["Normal Count"]):
        minSteepness, maxSteepness = Constants.AGENT_TYPE["normal"]["steepnessRange"]
        minTolerance, maxTolerance = Constants.AGENT_TYPE["normal"]["toleranceRange"]
        assignedSteepness = random.randint(minSteepness, maxSteepness)
        assignedTolerance = random.randint(minTolerance, maxTolerance)
        agents[idxRandom[idx]].setSteepnessTolerance(assignedSteepness, assignedTolerance)
        idx += 1
    for _ in range(agentsData["Stubborn Count"]):
        minSteepness, maxSteepness = Constants.AGENT_TYPE["stubborn"]["steepnessRange"]
        minTolerance, maxTolerance = Constants.AGENT_TYPE["stubborn"]["toleranceRange"]
        assignedSteepness = random.randint(minSteepness, maxSteepness)
        assignedTolerance = random.randint(minTolerance, maxTolerance)
        agents[idxRandom[idx]].setSteepnessTolerance(assignedSteepness, assignedTolerance)
        idx += 1

    # Set share propensity
    random.shuffle(idxRandom)
    idx = 0
    
    for _ in range(agentsData["Lurker Count"]):
        minPropensity, maxPropensity = Constants.AGENT_SHARE_PROPENSITY["lurker"]["propensityRange"]
        assignedPropensity = random.randint(minPropensity, maxPropensity)
        agents[idxRandom[idx]].setSharePropensity(assignedPropensity)
        idx += 1
    
    for _ in range(agentsData["Normal Sharer Count"]):
        minPropensity, maxPropensity = Constants.AGENT_SHARE_PROPENSITY["normal"]["propensityRange"]
        assignedPropensity = random.randint(minPropensity, maxPropensity)
        agents[idxRandom[idx]].setSharePropensity(assignedPropensity)
        idx += 1
    
    for _ in range(agentsData["Active Count"]):
        minPropensity, maxPropensity = Constants.AGENT_SHARE_PROPENSITY["active"]["propensityRange"]
        assignedPropensity = random.randint(minPropensity, maxPropensity)
        agents[idxRandom[idx]].setSharePropensity(assignedPropensity)
        idx += 1

    # Set active duration
    random.shuffle(idxRandom)
    idx = 0
    for _ in range(agentsData["Agent Count"]):
        minOnlineDuration, maxOnlineDuration = Constants.AGENT_DURATION["online"]
        minOfflineDuration, maxOfflineDuration = Constants.AGENT_DURATION["offline"]
        assignedOnlineDuration = random.randint(minOnlineDuration, maxOnlineDuration)
        assignedOfflineDuration = random.randint(minOfflineDuration, maxOfflineDuration)
        assignedStartingStatus = random.choice(["Online", "Offline"])
        assignedOfflineStartOffset = random.randint(0, (assignedOfflineDuration if assignedStartingStatus=="Online" else assignedOnlineDuration) - 1)
        agents[idxRandom[idx]].setActiveDuration(assignedOnlineDuration, assignedOfflineDuration, assignedStartingStatus, assignedOfflineStartOffset)
        idx += 1
    
    return agents

class Agent:
    def __init__(self, id, isNewsAgency=False):
        self.id = id
        self.beliefValue = 0

        self.steepness = 0
        self.tolerance = 0
        
        self.sharePropensity = 0.0 
        
        self.onlineDuration = 0
        self.offlineDuration = 0
        self.startingStatus = ""
        self.offlineStartOffset = 0
        
        self.followers = [] # followers instances
        self.feedQueue = []
        self.feedBuffer = set()
        self.interactionsDone = []
        self.sharedPosts = set()

        self.isNewsAgency = isNewsAgency

    def setBeliefValue(self, beliefValue) -> None:
        """value: -4 to 4 (int). if SBM, this is based on cluster"""
        self.beliefValue = beliefValue

    def setSteepnessTolerance(self, steepness, tolerance) -> None:
        """
        gullible: (1, 8-9)    15% of popu
        normal: (2, 4-7)      65% of popu
        stubborn: (3-4, 1-3)  20% of popu

        steepness value: 1 to 4 (int).
        tolerance value: 1 to 9 (int).
        """           
        self.steepness = steepness
        self.tolerance = tolerance

    def setSharePropensity(self, sharePropensity) -> None:
        """
        likeliness to share. 
        value: 0-100 (int) 
        90% - lurker, 9% - occasional contributors, 1% highly active
        """
        self.sharePropensity = sharePropensity

    def setActiveDuration(self, onlineDuration, offlineDuration, startingStatus, offlineStartOffset) -> None:
        """
        online/offline duration ranges from 1-8 hours.
        alternating between online and offline. cut if exceeds 48 hours
        """
        self.onlineDuration = onlineDuration
        self.offlineDuration = offlineDuration
        self.startingStatus = startingStatus
        self.offlineStartOffset = offlineStartOffset

    def getCognitiveResponse(self, post: "Post") -> float:
        if self.isNewsAgency:
            raise RuntimeError("Agent is a news agency, cannot get cognitive response.")


        postBeliefValue = post.getBeliefValue()
        postInterestValue = post.getInterestValue()
        agentSteepness = self.steepness
        agentTolerance = self.tolerance
        beliefDistance = abs(postBeliefValue - self.beliefValue)
        
        return 1/(1 + exp(agentSteepness*(beliefDistance-agentTolerance) - postInterestValue))

    def getDCCProbability(self, post: Post, time: int) -> float:
        "Get the Defensive Cognitive Cascade Probability given a Post."

        if self.isNewsAgency:
            raise RuntimeError("Agent is a news agency, cannot get DCC Probability.")

        postBeliefValue = post.getBeliefValue()
        postInterestValue = post.getInterestValue()
        agentSteepness = self.steepness
        agentTolerance = self.tolerance
        beliefDistance = abs(postBeliefValue - self.beliefValue)
        interestDecayConstant = 4

        dccProbability = self.getCognitiveResponse(post) * self.sharePropensity * (1 - (time/2880)**interestDecayConstant)
        return dccProbability
    
    def addPostToFeedBuffer(self, post, layer) -> None:
        "Appends post to agent's feed buffer."

        if self.isNewsAgency:
            raise RuntimeError("Agent is a news agency, cannot take in shared posts.")

        self.feedBuffer.add((post, layer))
            
    def addNewPostsToFeedQueue(self) -> None:
        "Adds posts from feed buffer to feed queue."
        if len(self.feedBuffer) == 0:
            return 

        for post, layer in sorted(list(self.feedBuffer), key=lambda x: (x[0].postingTime, x[0].postID)):
            self.feedQueue.append((post, layer))
                
        self.feedBuffer.clear() 

    def sharePost(self, post, layer, time, networkData) -> None:
        "Share post to all neighbors of the agent."
        for agent in self.followers:
            if not agent.hasPostBeenShared(post):
                agent.addPostToFeedBuffer(post, layer+1)
                if str(time) not in networkData["share_events"]:
                    networkData["share_events"][str(time)] = []
                networkData["share_events"][str(time)].append({
                    "source" : self.id,
                    "target" : agent.id,
                    "post_type" : ("misinformation" if post.isMisinformation else "regular")
                })
    
    def processFeed(self, networkData, time: int, simulationData) -> None:
        "Process all queued posts in feedQueue."

        if self.isNewsAgency:
            raise RuntimeError("Agent is a news agency, cannot take in shared posts.")

        for post, layer in self.feedQueue:
            if Constants.UPDATE_BELIEF_VALUE_BEFORE_SHARE:
                self.adjustBeliefValue(post)
            dccProbability = self.getDCCProbability(post, time)
            if BernoulliTrial(dccProbability):
                self.acceptPost(networkData, time, post, layer, simulationData)
            else:
                # generate PostInteraction with isShared==False
                self.interactionsDone.append(PostInteraction(      
                        time=time,
                        agent=self,
                        agentBelief=self.beliefValue,
                        post=post,
                        postInterestValue=post.getInterestValue(),
                        isShared=False,
                        layer=layer
                    )
                )

        self.feedQueue.clear()

    def acceptPost(self, networkData, time: int, post: "Post", layer: int, simulationData) -> None:
        "Does all needed processes once an agent accepts the contents of a post"
        
        if post.postID in self.sharedPosts:
            return
        
        if self.isNewsAgency:
            raise RuntimeError("Agent is a news agency, cannot take in shared posts.")

        self.sharePost(post, layer, time, networkData)
        # To do: all post interactions done by an agent will be stored in
        # a struct inherent to that agent, we can then just collect this later
        # on in order to do statistics
        self.interactionsDone.append(PostInteraction(      
                time=time,
                agent=self,
                agentBelief=self.beliefValue,
                post=post,
                postInterestValue=post.getInterestValue(),
                isShared=True,
                layer=layer
            )
        )
        simulationData[post.postID]["isMisinformation"] = post.isMisinformation
        simulationData[post.postID]["interactions"][time-post.postingTime] += 1


        self.sharedPosts.add(post.postID)

        # Update belief value after sharing
        if not Constants.UPDATE_BELIEF_VALUE_BEFORE_SHARE:
            self.adjustBeliefValue(post)

    def hasPostBeenShared(self, post: "Post") -> bool:
        "Checks if a post has already been shared by an agent"
        return post.postID in self.sharedPosts

    def isOnline(self, t: int) -> bool:
        "Returns current status of the agent."
        if t < self.offlineStartOffset:
            if self.startingStatus == "Online":
                return False
            else:
                return True
        elif self.startingStatus == "Online":
            return (t+self.offlineStartOffset) % (self.onlineDuration + self.offlineDuration) < self.onlineDuration
        elif self.startingStatus == "Offline":
            return not ((t+self.offlineStartOffset) % (self.onlineDuration + self.offlineDuration) < self.offlineDuration)
        
        raise RuntimeError("Agent is neither online or offline")

    def addFollower(self, otherIdx: "Agent") -> None:
        "Appends agent to followers."
        self.followers.append(otherIdx)

    def adjustBeliefValue(self, post: "Post") -> None:
        "Adjusts the agent belief value"

        if self.isNewsAgency:
            raise RuntimeError("Agent is a news agency, no belief value.")
        
        cognitiveResponse = self.getCognitiveResponse(post)
        oldBeliefValue = self.beliefValue
        newBeliefValue = -1
        if cognitiveResponse <= 0.33:
            # No change
            pass
        elif cognitiveResponse <= 0.44:
            # Minimal influence
            newBeliefValue = self.beliefValue + 1*(-1 if post.beliefValue < self.beliefValue else 1)
        elif cognitiveResponse <= 0.77:
            # Partial influence
            newBeliefValue = self.beliefValue + 2*(-1 if post.beliefValue < self.beliefValue else 1)
        elif cognitiveResponse <= 1:
            # Significant influence
            newBeliefValue = self.beliefValue + 3*(-1 if post.beliefValue < self.beliefValue else 1)
        else:
            raise RuntimeError("Updated belief value error")
        
        # if updated belief value exceeds post belief value, clamp agent belief value to post belief value
        if ((oldBeliefValue <= post.beliefValue and newBeliefValue > post.beliefValue) 
            or (oldBeliefValue >= post.beliefValue and newBeliefValue < post.beliefValue)
            ):
            newBeliefValue = post.beliefValue
        
        self.beliefValue = newBeliefValue 

    def classifyAgentType(self) -> str:
        """
        Returns 'gullible', 'stubborn', 'normal' based on 
        steepness and tolerance.
        """

        if self.isNewsAgency:
            raise RuntimeError("Agent is a news agency, no agent type.")

        for typeName, ranges in Constants.AGENT_TYPE.items():
            steepnessMin, steepnessMax = ranges["steepnessRange"]
            toleranceMin, toleranceMax = ranges["toleranceRange"]

            if ((steepnessMin <= self.steepness <= steepnessMax) and
               (toleranceMin <= self.tolerance <= toleranceMax)
            ):
                return typeName
        
        raise RuntimeError("Agent steepness and tolerance does not match any agent types")
            
    def classifyAgentBelief(self) -> str:
        """
        Returns 'red', 'centrist', or 'blue'
        """

        if self.isNewsAgency:
            raise RuntimeError("Agent is a news agency, no agent belief.")

        for beliefName, values in Constants.AGENT_BELIEF_TYPE.items():
            if self.beliefValue in values:
                return beliefName
        
        raise RuntimeError("Agent belief value not in range")