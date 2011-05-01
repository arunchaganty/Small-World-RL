"""
Implements the Macro Q-Learning Algorithm
"""

import OptionValueAgent
import collections
import operator
from Environment.OptionEnvironment import Option
import pdb
from numpy import random

class MacroQ(OptionValueAgent.OptionValueAgent):
    """
    Implements the Q-Learning
    """

    def update_Q(self, state, action, state_, action_, reward):
        """Update the Q function
        @state - old state (sequence)
        @action - old action
        @state_ - current state
        @action_ - current action
        @reward - reward (sequence)
        """

        if not state:
            return

        q = self.Q[state][action]
        # Check if the state is a list for actions
        if isinstance( action, Option ):
            state_ = state_[-1][0]
            k = len(reward)
            reward = sum( [ self.gamma**i * reward[i] for i in xrange(k) ] )
        else:
            k = 1

        if not state_:
            q += self.alpha * (reward - q)
        else:
            q_ = max(self.Q[state_].values())
            q += self.alpha * (reward + (self.gamma**k) * q_ - q)

        self.Q[state][action] = q

