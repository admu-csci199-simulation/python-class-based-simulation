from Random import rng as random


EPSILON = 1e-9

def BernoulliTrial(probability, maxRange=None):
    """
    Simulate a Bernoulli trial.
    - If probability is a float in [0,1], performs a float-based trial.
    - If probability and maxRange are ints, performs an int-based trial.
    """
    # Case 1: float probability (0 <= p <= 1)
    if isinstance(probability, float) and maxRange is None:
        random_sample = random.random()
        return random_sample < probability

    # Case 2: int probability with maxRange
    elif isinstance(probability, int) and isinstance(maxRange, int):
        random_sample = random.randint(0, maxRange)  # inclusive
        return random_sample < probability

    else:
        raise TypeError("Invalid arguments. Use (float) or (int, int).")
