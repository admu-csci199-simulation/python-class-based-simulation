class PostInteraction:
    def __init__(self, time, agent, agentBelief, post, postInterestValue):
        self.time = time
        self.agent = agent
        self.agentBelief = agentBelief # agent belief at time t
        self.post = post
        self.postInterestValue = postInterestValue # post interest value at time t

    # create function that generates graphs here