# This class is generated after an agent accepted a post via DDC function.
# Receiving a post and rejecting to share it does NOT count as an agent accepting a post.
class PostInteraction:
    def __init__(self, time, agent, agentBelief, post, postInterestValue, isShared, layer):
        self.time = time
        self.agent = agent
        self.agentBelief = agentBelief # agent belief at time t
        self.post = post
        self.postInterestValue = postInterestValue # post interest value at time t
        self.isShared = isShared
        layer=layer

    def getNthLayer(self):
        return self.time - self.post.postingTime