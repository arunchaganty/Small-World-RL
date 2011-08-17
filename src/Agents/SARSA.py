"""
RL Framework
Authors: Arun Chaganty
Implements SARSA
"""

from Agent import *

class SARSA(ValueAgent):
    """
    Implements SARSA
    """

    def update_Q(self, state, action, state_, action_, reward):
        if not state:
            return

        q = self.get_value( state, action )

        if not state_:
            q += self.alpha * (reward - q)
        else:
            q_ = self.get_value( state_, action_ )
            q += self.alpha * (reward + self.gamma * q_ - q)

        self.set_value( state, action, q )

