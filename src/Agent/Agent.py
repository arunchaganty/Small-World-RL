"""
Agent Base Class
"""

class Agent:
    def __init__(self):
        pass

    def act(self, state, actions, reward, episode_ended):
        raise NotImplemented()

