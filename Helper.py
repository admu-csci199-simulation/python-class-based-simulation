import random


EPSILON = 1e-9

def BernoulliTrial(probability: float):
    "Given a probability, simulate probability result using a trial"
    randomSample = random.random()
    return probability - randomSample <= EPSILON
