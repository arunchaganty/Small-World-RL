"""
Implements the Intra Option Q-Learning Algorithm
"""

import OptionValueAgent
import collections
import operator
from Environment.OptionEnvironment import Option, DeterministicOption
import pdb
from numpy import random

class IntraOptionQ(OptionValueAgent.OptionValueAgent):
    """
    Implements the Intra Option Q-Learning Algorithm
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

        def do_update( st, a, st_, a_, r ):
            q = self.Q[st][a]
            if st_:
                q_ = self.Q[st_].get(a_,0)
            else:
                q_ = 0
            q += self.alpha * (r + self.gamma * q_ - q)
            self.Q[st][a] = q

        if isinstance( action, Option ):
            # HACK
            if (len(reward) == len( state_ ) ):
                state_ = tuple(state_) + ((None, None, None),)
            # Traverse the state sequence 
            for i in xrange( len( reward ) ):
                # Find all the updatable options and actions
                st, a, al = state_[i]
                st_, a_, al_ = state_[i+1]

                if st_:
                    max_a = max( ( x for x in al_ if not isinstance(x, Option) ), key = lambda x: self.Q[st_].get(x,0) )
                else:
                    max_a = None

                # Q-update
                do_update( st, a, st_, max_a, reward[i] )
                for x in al:
                    if isinstance(x, DeterministicOption) and x.can_start(st) and x.policy[st] == a: 
                        do_update( st, x, st_, x, reward[i] )
        else:
            do_update( state, action, state_, action_, reward )

    def __option_has_action( self, state, a, action ):
        if isinstance( a, DeterministicOption ):
            return a.policy[ state ] == action
        else:
            raise NotImplemented()
        
