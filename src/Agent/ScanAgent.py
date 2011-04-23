"""
Implements a scanning agent.
"""

import Agent

PLAYER_X = -1
PLAYER_N = 0
PLAYER_O = 1

class ScanAgent(Agent.Agent):
    """
    Implements a simple scanning agent.
    """
    def __init__(self):
        Agent.Agent.__init__(self)

    def act(self, state, actions, rewards, episode_ended):
        """Chooses the first free cell, scanning row wise"""
        board = state
        for row in xrange(len(board)):
            for col in xrange(len(board[row])):
                if board[row][col] == PLAYER_N:
                    return (row, col)

