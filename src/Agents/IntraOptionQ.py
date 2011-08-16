"""
Implements the Intra Option Q-Learning Algorithm
"""

from Agent import *
from Environment import *

import pdb
import numpy as np

class IntraOptionQ(OptionValueAgent):
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
            if st_:
                # Find the highest value primitive action
                q_ = max( ( q_ for (a_, q_) in self.Q[st_] if not isinstance( a_, Option ) ) )
            else:
                # Happens only at end of episodes
                q_ = 0

            # Q-update of primitive action
            q = self.get_value( st, a )
            q += self.alpha * (r + self.gamma * q_ - q)
            self.set_value( st, a, q )

            # Update all options that have this action in their policy as
            # well
            for (o, q) in self.Q[ st ]:
                if isinstance( o, Option ) and any( ( a_ for (a_,pr) in o.pi[ st ] if a_ == a) ):
                    q = self.get_value( st, o )
                    if st_:
                        q_ = ( 1 - o.B( st ) ) * self.get_value( st_, o ) + o.B( st ) * max( ( q_ for (a_, q_) in self.Q[st_] ) )
                    else:
                        q_ = 0
                    q += self.alpha * (r + self.gamma * q_ - q)
                    self.set_value( st, o, q )

        if isinstance( action, Option ):
            # Traverse the state sequence 
            for i in xrange( len( reward ) ):
                # Find all the updatable options and actions
                st, a = state_[ i ]
                st_, a_ = state_[i+1]
                r = reward[ i ]
                do_update( st, a, st_, a_, r )
        else:
            do_update( state, action, state_, action_, reward )

