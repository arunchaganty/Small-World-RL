"""
Implements the random agent.
"""

from Agent import *
import random

class RandomAgent(Agent):
    def act( self, state, reward, episode_ended ):
        action = random.choice( self.Q[ state ] )
        return action

