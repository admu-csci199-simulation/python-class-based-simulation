from math import e as EULER 
from Post import Post
from Helper import BernoulliTrial
from PostInteraction import PostInteraction
import Constants
import random


def generateAgents(seed=Constants.GRAPH_SEED):
    random.seed(seed)
    agents = [Agent() for i in range(Constants.N_AGENTS)]

    # Set belief values
    # assumption: cluster sizes are equal
    for agentIdx in range(Constants.N_AGENTS):
        clusterIdx = agentIdx//(Constants.N_AGENTS//3)
        minBeliefValue, maxBeliefValue = Constants.BELIEF_VALUES[clusterIdx]
        assignedBeliefValue = random.randint(minBeliefValue, maxBeliefValue)
        agents[agentIdx].setBeliefValue(assignedBeliefValue)
    
    # Set steepness tolerance
    idxRandom = [i for i in range(Constants.N_AGENTS)] # randomized agents index 
    random.shuffle(idxRandom)
    idx = 0
    for agentType, values in Constants.AGENT_TYPE.items():
        for _ in range(values["count"]):
            minSteepness, maxSteepness = values["steepnessRange"]
            minTolerance, maxTolerance = values["toleranceRange"]
            assignedSteepness = random.randint(minSteepness, maxSteepness)
            assignedTolerance = random.randint(minTolerance, maxTolerance)
            agents[idxRandom[idx]].setSteepnessTolerance(assignedSteepness, assignedTolerance)
            idx += 1

    # Set share propensity
    random.shuffle(idxRandom)
    idx = 0
    for userType, values in Constants.AGENT_SHARE_PROPENSITY.items():
        for _ in range(values["count"]):
            minPropensity, maxPropensity = values["propensityRange"]
            assignedPropensity = random.randint(minPropensity, maxPropensity)
            agents[idxRandom[idx]].setSharePropensity(assignedPropensity)
            idx += 1

    # Set active duration
    random.shuffle(idxRandom)
    idx = 0
    for _ in range(Constants.N_AGENTS):
        minOnlineDuration, maxOnlineDuration = Constants.AGENT_DURATION["online"]
        minOfflineDuration, maxOfflineDuration = Constants.AGENT_DURATION["offline"]
        assignedOnlineDuration = random.randint(minOnlineDuration, maxOnlineDuration)
        assignedOfflineDuration = random.randint(minOfflineDuration, maxOfflineDuration)
        assignedStartingStatus = "Online" # ASSUMED
        agents[idxRandom[idx]].setActiveDuration(assignedOnlineDuration, assignedOfflineDuration, assignedStartingStatus)
        idx += 1
    
    return agents

class Agent:
    def __init__(self):
        self.beliefValue = 0

        self.steepness = 0
        self.tolerance = 0
        
        self.sharePropensity = 0.0 
        
        self.onlineDuration = 0
        self.offlineDuration = 0
        self.startingStatus = ""
        
        self.followers = [] # followers instances
        self.feedQueue = []
        self.feedBuffer = set()
        self.interactionsDone = []

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

    def setActiveDuration(self, onlineDuration, offlineDuration, startingStatus) -> None:
        """
        online/offline duration ranges from 1-8 hours.
        alternating between online and offline. cut if exceeds 48 hours
        """
        self.onlineDuration = onlineDuration
        self.offlineDuration = offlineDuration
        self.startingStatus = startingStatus

    def getDCCProbability(self, post: Post) -> float:
        "Get the Defensive Cognitive Cascade Probability given a Post."
        postBeliefValue = post.getBeliefValue()
        postInterestValue = post.getInterestValue()
        agentSteepness = self.steepness
        agentTolerance = self.tolerance
        beliefDistance = abs(postBeliefValue - self.beliefValue)
        
        dccProbability = 1/(1 + EULER**(agentSteepness*(beliefDistance-agentTolerance) - postInterestValue))
        return dccProbability
    
    def addPostToFeedBuffer(self, post) -> None:
        "Appends post to agent's feed buffer."
        self.feedBuffer.add(post)
            
    def addNewPostsToFeedQueue(self) -> None:
        "Adds posts from feed buffer to feed queue."
        if len(self.feedBuffer) == 0:
            return 

        for post in self.feedBuffer:
            self.feedQueue.append(post)
                
        self.feedBuffer.clear() 

    def sharePost(self, post) -> None:
        "Share post to all neighbors of the agent."
        for agent in self.followers:
            agent.addPostToFeedBuffer(post)
    
    def processFeed(self, time: int) -> None:
        "Process all queued posts in feedQueue."
        for post in self.feedQueue:
            dccProbability = self.getDCCProbability(post)
            if BernoulliTrial(dccProbability):
                self.acceptPost(time, post)

        self.feedQueue.clear()

    def acceptPost(self, time: int, post: "Post") -> None:
        "Does all needed processes once an agent accepts the contents of a post"
        self.sharePost(post)
        # To do: all post interactions done by an agent will be stored in
        # a struct inherent to that agent, we can then just collect this later
        # on in order to do statistics
        self.interactionsDone.append(PostInteraction(      
                time,
                self,
                self.beliefValue,
                post,
                post.getInterestValue()
            )
        )
        self.adjustBeliefValue(post.getBeliefValue())

    def isOnline(self, t: int) -> bool:
        "Returns current status of the agent."
        if self.startingStatus == "Online":
            return t % (self.onlineDuration + self.offlineDuration) <= self.onlineDuration
        elif self.startingStatus == "Offline":
            return t % (self.onlineDuration + self.offlineDuration) <= self.offlineDuration
        assert(False)

    def addFollower(self, otherIdx: int) -> None:
        "Appends agent to followers."
        self.followers.append(otherIdx)

    def adjustBeliefValue(self, postBeliefValue: int) -> None:
        "Adjusts the agent belief value"
        beliefDiff = abs(postBeliefValue - self.beliefValue)
        newBeliefValue = 0
        if beliefDiff <= 2:
            newBeliefValue = postBeliefValue
        elif beliefDiff <= 5:
            newBeliefValue = self.beliefValue + 2*(-1 if postBeliefValue < self.beliefValue else 1)
        elif beliefDiff <= 8:
            newBeliefValue = self.beliefValue + 1*(-1 if postBeliefValue < self.beliefValue else 1)
        else:
            assert(False) # unexpected belief difference 
        self.beliefValue = newBeliefValue
        self.beliefValue = newBeliefValue 

    def classifyAgentType(self) -> str:
        """
        Returns 'gullible', 'stubborn', 'normal' based on 
        steepness and tolerance.
        """
        for typeName, ranges in Constants.AGENT_TYPE.items():
            steepnessMin, steepnessMax = ranges["steepnessRange"]
            toleranceMin, toleranceMax = ranges["toleranceRange"]

            if (steepnessMin <= self.steepness <= steepnessMax) and \
               (toleranceMin <= self.tolerance <= toleranceMax):
                return typeName
            
    def classifyAgentBelief(self) -> str:
        """
        Returns 'red', 'centrist', or 'blue'
        """
        for beliefName, values in Constants.AGENT_BELIEF_TYPE.items():
            if self.beliefValue in values:
                return beliefName
        