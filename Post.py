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