"""
Generic Value Based agent
"""

import pdb
import Agent
import collections
from numpy import random

def choose( lst ):
    return lst[random.randint(len(lst))]

class ValueAgent(Agent.Agent):
    """
    Generic Value-based agent
    """

    Q = {}

    def __init__(self, gamma = 0.9, alpha = 0.1, e = 0.05, rate = 0.001):
        Agent.Agent.__init__(self)
        self.gamma = gamma
        self.alpha = alpha
        self.e = e
        self.rate = rate
        self.old_state = None
        self.old_action = None

    def update_Q(self, state, action, state_, action_, reward):
        """Update the Q function
        @state - old state
        @action - old action
        @state_ - current state
        @action_ - current action
        @reward - reward
        """
        raise NotImplemented()

    def act(self, state, actions, reward, episode_ended):
        # epsilon-greedy
        if not self.Q.has_key(state):
            self.Q[state] = {}
        for action in actions:
            if not self.Q[state].has_key(action):
                self.Q[state][action] = 0

        # Explore
        if random.random() < self.e:
            action = choose( actions )
        # Exploit
        else:
            max_value = max( self.Q[state].values() )
            action = choose( tuple( a for a in actions if self.Q[state][a] == max_value ) )

        # Update actions
        if episode_ended:
            self.update_Q(self.old_state, self.old_action, None, None, reward)
            # Cool e every episode
            self.e = self.e * (1-self.rate)
        else:
            self.update_Q(self.old_state, self.old_action, state, action, reward)

        self.old_state = state
        self.old_action = action

        return action

