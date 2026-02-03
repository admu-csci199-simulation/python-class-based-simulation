import random
import Constants

def generatePost(postID, beliefValue=None, originalPoster=None, postingTime=None, interestValue=None, postTopic=None):
    if beliefValue is None:
        beliefValue = random.randint(-4, 4)
    if originalPoster is None:
        originalPoster = random.randint(0, Constants.N_AGENTS-1)
    if postingTime is None:
        postingTime = 0
    if interestValue is None:
        interestValue = 4
    if postTopic is None:
        postTopic = random.randint(0, Constants.N_TOPICS-1)
    
    return Post(postID, beliefValue, interestValue, postingTime, originalPoster, postTopic)

class Post:
    def __init__(self, postID, beliefValue, interestValue, postingTime, originalPoster, postTopic):
        self.postID = postID
        self.beliefValue = beliefValue
        self.interestValue = interestValue
        self.postingTime = postingTime
        self.originalPoster = originalPoster
        self.postTopic = postTopic
        self.interactions = []
    
    def __repr__(self):
        return f"post_{self.postID}"

    def getBeliefValue(self) -> int:
        return self.beliefValue
    
    def getInterestValue(self) -> int:
        return self.interestValue