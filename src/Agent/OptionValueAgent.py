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

    def act(self, state, actions, reward, episode_ended):
        if isinstance( self.old_action, Option ):
            state_ = state[-1][0]
        else:
            state_ = state
        # epsilon-greedy
        if not self.Q.has_key(state_):
            self.Q[state_] = {}
        for action in actions:
            if not self.Q[state_].has_key(action):
                self.Q[state_][action] = 0

        # Explore
        if random.random() < self.e:
            action = choose( actions )
        # Exploit
        else:
            max_value = max(self.Q[state_].values())
            action = choose( [ a for a in actions if self.Q[state_][a] == max_value ] )

        # Update actions
        if episode_ended:
            self.update_Q(self.old_state, self.old_action, None, None, reward)
            # Cool e every episode
            self.e = self.e * (1-self.rate)
        else:
            self.update_Q(self.old_state, self.old_action, state, action, reward)

        self.old_state = state_
        self.old_action = action

        return action

