"""
RL Framework
Authors: Arun Chaganty
Implements the Q-Learning algorithm
"""

from Agent import *

class QLearning(ValueAgent):
    """
    Q-Learning algorithm
    """

    def update_Q(self, state, action, state_, action_, reward):
        if not state:
            return

        q = self.get_value( state, action )

        if not state_:
            q += self.alpha * (reward - q)
        else:
            q_ = max( ( pr for (a_,pr) in self.Q[state_] ) )
            q += self.alpha * (reward + self.gamma * q_ - q)

        self.set_value( state, action, q )

