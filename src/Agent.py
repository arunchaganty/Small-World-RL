"""
RL Framework
Authors: Arun Chaganty
Agent Base Classes; Represented by an MDP
"""

import numpy as np
import random

from Environment import *

class Agent:
    Q = []
    def __init__(self, Q):
        self.Q = Q

    def act( self, state, reward, episode_ended ):
        raise NotImplemented()

    def greedy_policy( self ):
        S = len( self.Q )

        pi = {}
        for s in xrange( S ):
            # Choose greedy action
            actions = self.Q[ s ]
            max_value = max( ( pr for (a,pr) in actions ) )
            a = random.choice( tuple( a for (a,pr) in actions if pr == max_value ) )
            pi[s] = (a,1.0)
        return pi

class ValueAgent( Agent ):
    old_state = None
    old_action = None
    e = 0.01
    alpha = 0.1
    gamma = 0.9
    rate = 0.99

    def __init__(self, Q, e = 0.01, alpha = 0.1, gamma = 0.9, rate = 0.99 ):
        Agent.__init__( self, Q )

        self.Q = []
        for A in Q:
            self.Q.append( [ (a,0) for a in A ] )

        self.e = e
        self.alpha = alpha
        self.gamma = gamma
        self.rate = rate
        self.old_state = None
        self.old_action = None

    def act( self, state, reward, episode_ended ):

        # Explore
        actions = self.Q[ state ]
        if random.random() < self.e:
            action = random.choice( tuple( a for (a,pr) in actions ) )
        # Exploit
        else:
            max_value = max( ( pr for (a,pr) in actions ) )
            action = random.choice( tuple( a for (a,pr) in actions if pr == max_value ) )

        # Update actions
        if episode_ended:
            self.update_Q( self.old_state, self.old_action, None, None, reward )
            self.e = self.e * (1 - self.rate)
        else:
            self.update_Q( self.old_state, self.old_action, state, action, reward )

        self.old_state = state
        self.old_action = action

        return action

    def update_Q( self, state, action, state_, action_, reward ):
        raise NotImplemented()


    def get_idx( self, state, action ):
        actions = self.Q[state]
        for i in xrange( len( actions ) ):
            if action == actions[ i ][ 0 ]: 
                return i
        raise ValueError()

    def get_value( self, state, action ):
        try:
            a, q = self.Q[ state ][ self.get_idx( state, action ) ]
        except ValueError:
            q = 0
        return q

    def set_value( self, state, action, value ):
        self.Q[ state ][ self.get_idx( state, action ) ] = (action, value)

class OptionValueAgent( ValueAgent ):

    def act( self, state, reward, episode_ended ):

        # Revisit the history and update using it as well
        if isinstance( self.old_action, Option ):
            state_  = state[-1][0] 
        else:
            state_ = state
        
        # Explore
        actions = self.Q[ state_ ]
        if random.random() < self.e:
            action = random.choice( tuple( a for (a,pr) in actions ) )
        # Exploit
        else:
            max_value = max( ( pr for (a,pr) in actions ) )
            action = random.choice( tuple( a for (a,pr) in actions if pr == max_value ) )

        # Update actions
        if episode_ended:
            # In the case of options, send along the old state list
            if isinstance( self.old_action, Option ):
                # Replace the last state with a None one (end of epsiode)
                state = state[:-1] + [(None, None),]
                self.update_Q( self.old_state, self.old_action, state, None, reward )
            else:
                self.update_Q(self.old_state, self.old_action, None, None, reward)
            self.e = self.e * (1 - self.rate)
        else:
            self.update_Q( self.old_state, self.old_action, state, action, reward )

        self.old_state = state_
        self.old_action = action

        return action

    def update_Q( self, state, action, state_, action_, reward ):
        raise NotImplemented()

