"""
Generic Value Based agent for Options
"""

import pdb
import Agent, ValueAgent
from Environment.OptionEnvironment import Option
import collections
from numpy import random

def choose( lst ):
    return lst[random.randint(len(lst))]

class OptionValueAgent(ValueAgent.ValueAgent):
    """
    Generic Value-based agent
    """

    def register_state_action( self, state, actions ):
        if not self.Q.has_key( state ):
            self.Q[state] = {}
        for action in actions:
            if not self.Q[state].has_key(action):
                self.Q[state][action] = 0

    def act(self, state, actions, reward, episode_ended):
        if isinstance( self.old_action, Option ):
            for st, a, as_ in state:
                self.register_state_action( st, as_ )
            state_ = state[-1][0]
        else:
            self.register_state_action( state, actions )
            state_ = state

        # Explore
        if random.random() < self.e:
            action = choose( actions )
        # Exploit
        else:
            max_value = max( [ self.Q[state_][a] for a in actions ] )
            #pdb.set_trace()
            action = choose( [ a for a in actions if self.Q[state_][a] == max_value ] )

        # Update actions
        if episode_ended:
            # In the case of options, send along the old state list
            if isinstance( self.old_action, Option ):
                self.update_Q(self.old_state, self.old_action, state[:-1], None, reward)
            else:
                self.update_Q(self.old_state, self.old_action, None, None, reward)
            # Cool e every episode
            self.e = self.e * (1-self.rate)
        else:
            self.update_Q(self.old_state, self.old_action, state, action, reward)

        self.old_state = state_
        self.old_action = action

        return action

