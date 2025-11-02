import random
import Constants
from Post import Post


def generatePost(beliefValue=None, originalPoster=None, postingTime=None):
    if beliefValue is None:
        beliefValue = random.randint(-3, 3)
    if originalPoster is None:
        originalPoster = random.randint(0, Constants.N_AGENTS-1)
    if postingTime is None:
        postingTime = 0
    interestValue = random.randint(-3, 3)
    
    return Post(beliefValue, interestValue, postingTime, originalPoster)
