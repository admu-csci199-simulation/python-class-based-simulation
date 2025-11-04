import random
import Constants


def generatePost(beliefValue=None, originalPoster=None, postingTime=None):
    if beliefValue is None:
        beliefValue = random.randint(-3, 3)
    if originalPoster is None:
        originalPoster = random.randint(0, Constants.N_AGENTS-1)
    if postingTime is None:
        postingTime = 0
    interestValue = random.randint(-3, 3)
    
    return Post(beliefValue, interestValue, postingTime, originalPoster)

class Post:
    def __init__(self, beliefValue, interestValue, postingTime, originalPoster):
        self.beliefValue = beliefValue
        self.interestValue = interestValue
        self.postingTime = postingTime
        self.originalPoster = originalPoster
        self.interactions = []
    
    def getBeliefValue(self) -> int:
        return self.beliefValue
    
    def getInterestValue(self) -> int:
        return self.interestValue