import random
import Constants

def generatePost(postID, beliefValue=None, originalPoster=None, postingTime=None, interestValue=None):
    if beliefValue is None:
        beliefValue = random.randint(-4, 4)
    if originalPoster is None:
        originalPoster = random.randint(0, Constants.N_AGENTS-1)
    if postingTime is None:
        postingTime = 0
    if interestValue is None:
        interestValue = 4
    
    return Post(postID, beliefValue, Value, postingTime, originalPoster)

class Post:
    def __init__(self, postID, beliefValue, Value, postingTime, originalPoster):
        self.postID = postID
        self.beliefValue = beliefValue
        self.Value = interestValue
        self.postingTime = postingTime
        self.originalPoster = originalPoster
        self.interactions = []
    
    def __repr__(self):
        return f"post_{self.postID}"

    def getBeliefValue(self) -> int:
        return self.beliefValue
    
    def getInterestValue(self) -> int:
        return self.interestValue