"""
Implements a policy-gradient agent
"""

import Agent
import numpy as np
import scipy as sp
import scipy.maxentropy
from numpy import random

class RandomDistribution:
    """Represents a generic random distribution"""
    def __init__(self, pdf):
        self.s = "Random"
        """Creates an arbitrary random distribution"""
        self.pdf = np.array(pdf)
        # Some pdf values may be zero - in which case skip their values
        self.cdf = self.__cdf(pdf)

    def __cdf(self, pdf):
        """Computes the CDF"""
        values = np.array(pdf)
        for i in xrange(1, len(values)):
            values[i] += values[i-1]
        return values

    def sample(self):
        """ Sample from the distribution according to the pdf."""
        r = random.random()

        for (i, p) in enumerate(self.pdf):
            if r <= p:
                return i
            r -= p
        return len(self.pdf) - 1

class GibbsDistribution(RandomDistribution):
    def __init__(self, pdf, T):
        """Creates a gibbs random distribution"""
        # Consider only the log values for now
        pdf = np.array(pdf)

        # Shift by the minimum value of the pdf to prevent overflow
        log_pdf = (pdf - min(pdf))/float(T)
        # Compute logsum
        log_Z = scipy.maxentropy.logsumexp( log_pdf )
        # Then,
        pdf = np.exp( log_pdf - log_Z )

        RandomDistribution.__init__(self, pdf)
        # print "Initialising - ", pdf
        self.s = "Gibbs"


def state_(state):
    """Create a hashable by converting to a tuple"""
    return tuple( [ tuple( row ) for row in state ] )

def count_moves(board):
    """Count the number of moves that have been played on the board
    @returns movedFirst?, moveNumber
    """
    count = abs(board).sum()
    return count/2

class PolicyGradient(Agent.Agent):
    r"""
    Implements a policy-gradient agent
    This performs the following update:
    p(s, a_i) <- p(s, a_i) + \alpha ( 1 - R_t \sum_j ... )
    Assuming a Gibbs-Boltzmann Distribution
    """

    theta = {}
    trajectory = []

    def __init__(self, beta = 0.1, T = 0.1):
        Agent.Agent.__init__(self)
        self.trajectory = []
        self.move_count = 0
        self.beta = beta
        self.T = T

    def print_policy(self, state):
        actions = self.theta[state]
        dist = GibbsDistribution( actions.values(), self.T)
        actions_ = dict( zip( actions.keys(), dist.pdf ) )

        print "=========================="
        for i in xrange(3):
            for j in xrange(3):
                print "|",
                if state[i][j] == 1:
                    print "  O  |",
                elif state[i][j] == -1:
                    print "  X  |",
                else:
                    print "%.2f |"%(actions[(i,j)]),
            print ""
        print "=========================="

        print "=========================="
        for i in xrange(3):
            for j in xrange(3):
                print "|",
                if state[i][j] == 1:
                    print "  O  |",
                elif state[i][j] == -1:
                    print "  X  |",
                else:
                    print "%.2f |"%(actions_[(i,j)]),
            print ""
        print "=========================="

    def init_state(self, hashed_state, actions):
        self.theta[hashed_state] = {}
        for action in actions:
            self.theta[hashed_state][action] = 0

    def detect_episode_boundary(self, state):
        count = self.move_count
        self.move_count = count_moves(state)
        return count > self.move_count

    def update_theta(self, reward):
        """Updates the action preferences (theta_i)"""

        self.trajectory.reverse()
        for hashed_state, chosen_action in self.trajectory:
            action_values = self.theta[hashed_state].items()
            dist = GibbsDistribution( [v for (k,v) in action_values], self.T)

            for i in xrange( len(action_values) ):
                possible_action = action_values[i][0]
                val = dist.pdf[i]
                if possible_action == chosen_action:
                    update = self.beta * reward * (1 - val)
                else:
                    update = self.beta * reward * (-val)
                self.theta[hashed_state][possible_action] += update

    def act(self, state, actions, reward, episode_ended):
        # Hash the state
        hashed_state = state_(state)
        # Detect if the episode has finished
        if episode_ended and len(self.trajectory) > 0:
            self.update_theta(reward)
            self.trajectory = []

        if not self.theta.has_key(hashed_state):
            self.init_state(hashed_state, actions)
        dist = GibbsDistribution( self.theta[hashed_state].values(), self.T)
        action = self.theta[hashed_state].keys()[ dist.sample() ]
        self.trajectory.append((hashed_state, action))
        #self.print_policy( hashed_state )

        return action

