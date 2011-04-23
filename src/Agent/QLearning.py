"""
Implements the Q-Learning
"""

import ValueAgent
from numpy import random

class QLearning(ValueAgent.ValueAgent):
    """
    Implements the Q-Learning
    """

    def update_Q(self, state, action, state_, action_, reward):
        if not state:
            return

        q = self.Q[state][action]
        if not state_:
            q += self.alpha * (reward - q)
        else:
            q_ = max(self.Q[state_].values())
            q += self.alpha * (reward + self.gamma * q_ - q)

        self.Q[state][action] = q

