"""
Implements the Macro Q-Learning Algorithm
"""

from Agent import *
from Environment import *
from numpy import random

class MacroQ(OptionValueAgent):
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

        q = self.get_value( state, action )

        # If action is an option,
        if isinstance( action, Option ):
            if state_:
                state_ = state_[-1][0]
            k = len(reward)
            reward = np.sum( np.exp( self.gamma * np.ones( k ), np.arange( k ) ) * np.array( reward ) )
        else:
            k = 1

        if not state_:
            q += self.alpha * (reward - q)
        else:
            q_ = max( ( pr for (a_,pr) in self.Q[state_] ) )
            q += self.alpha * (reward + np.power(self.gamma, k) * q_ - q)

        self.set_value( state, action, q )

