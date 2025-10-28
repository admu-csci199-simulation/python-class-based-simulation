from math import e as EULER
from Post import Post
from Helper import BernoulliTrial

def generateAgents(
        n=Constants.N_AGENTS,
        agentType = Constants.agentType
    ):
    pass

class Agent:
    def __init__(self, beliefValue, steepness, tolerance, sharePropensity, onlineDuration, offlineDuration, startingStatus):
        # value: -4 to 4 (int). if SBM based on cluster
        self.beliefValue = beliefValue

        # gullible: (1, 8-9)    15% of popu
        # normal: (2, 4-7)      65% of popu
        # stubborn: (3-4, 1-3)  20% of popu

        # value: 1 to 4 (int).   
        self.steepness = steepness
        # value: 1 to 9 (int).  
        self.tolerance = tolerance

        # likeliness to share. 
        # value: 0-100 (int) 
        # 90% - lurker, 9% - occasional contributors, 1% highly active
        self.sharePropensity = sharePropensity 

        # ranging (1 to 8 hours)
        self.onlineDuration = onlineDuration
        self.offlineDuration = offlineDuration
        self.startingStatus = startingStatus
        
        self.followers = []
        self.feedQueue = []

    def getDCCProbability(self, post: Post) -> float:
        "Get the Defensive Cognitive Cascade Probability given a Post."
        postBeliefValue = post.getBeliefValue()
        postInterestValue = post.getInterestValue()
        agentSteepness = self.steepness
        agentTolerance = self.tolerance
        beliefDistance = abs(postBeliefValue - self.beliefValue)
        
        dccProbability = 1/(1 + EULER**(agentSteepness*(beliefDistance-agentTolerance) - postInterestValue))
        return dccProbability
    
    def addPostToFeed(self, post):
        "Appends post to agent's feed."
        self.feedQueue.append(post)
    
    def sharePost(self, post):
        "Share post to all neighbors of the agent."
        for agent in self.followers:
            agent.addPostToFeed(post)
    
    def processFeed(self):
        "Process all queued posts in feedQueue."
        for post in self.feedQueue:
            dccProbability = self.getDCCProbability(post)
            if BernoulliTrial(dccProbability):
                self.sharePost(post)
        self.feedQueue.clear()

    def getCurrentStatus(self, t: int):
        "Returns current status of the agent."
        if self.startingStatus == "Online":
            return t % (self.onlineDuration + self.offlineDuration) <= self.onlineDuration
        elif self.startingStatus == "Offline":
            return t % (self.onlineDuration + self.offlineDuration) <= self.offlineDuration
        assert(False)

    def addFollower(self, other: "Agent"):
        "Appends agent to followers."
        self.followers.append(other)