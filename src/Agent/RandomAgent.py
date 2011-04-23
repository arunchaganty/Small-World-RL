"""
Implements the random agent.
"""

from Agent import Agent
from numpy import random

class RandomAgent(Agent):
    def __init__(self):
        pass

    def act(self, state, actions, reward, episode_ended):
        return actions[ random.randint(len(actions)) ]

