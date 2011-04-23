"""
Implements the optimal agent
"""

import Agent
from numpy import random
import numpy as np

PLAYER_X = -1
PLAYER_N = 0
PLAYER_O = 1

CORNERS = [(0, 0), (0, 2), (2, 0), (2, 2)]
CENTER = (1, 1)
EDGE_CENTERS = [(0, 1), (1, 0), (1, 2), (2, 1)]
EMPTY = (-1, -1)

def free_corners(actions):
    """Get the free corners"""
    return list(set(actions).intersection(CORNERS))
def free_edges(actions):
    """Get the free edges"""
    return list(set(actions).intersection(EDGE_CENTERS))
def free_center(actions):
    """Get the free center"""
    if CENTER in actions:
        return CENTER
    else:
        return None

def has_corner(board, player):
    """Checks if player has any corner"""
    return len(get_corners(board, player)) > 0
def has_edge(board, player):
    """Checks if player has any edge"""
    return len(get_edges(board, player)) > 0
def has_center(board, player):
    """Checks if player has the center"""
    return board[CENTER[0]][CENTER[1]] == player

def get_corners(board, player):
    """Gets the corners the player has"""
    return [ (i, j) for (i, j) in CORNERS if board[i][j] == player ]
def get_edges(board, player):
    """Gets the edges the player has"""
    return [ (i, j) for (i, j) in EDGE_CENTERS if board[i][j] == player ]

def flip(pos):
    """Flips a position"""
    return (2-pos[0], 2-pos[1])

def choose(lst):
    """Choose an element from a list"""
    return lst[random.randint(len(lst))]

def tuple_add(t1, *t2):
    """Add two tuples"""
    for t in t2:
        t1 = tuple([ x + y for (x,y) in zip(t1, t) ])
    return t1

class OptimalAgent(Agent.Agent):
    """ Implements the optimal agent """
    def __init__(self):
        Agent.Agent.__init__(self)

    def count_moves(self, board):
        """Count the number of moves that have been played on the board
        @returns movedFirst?, moveNumber
        """
        count = abs(board).sum()
        return count/2

    def first_player(self, board):
        """ Checks if the agent is the first player """
        count = abs(board).sum()
        return count % 2 == 0

    def mark(self, board):
        """ Returns the mark of the agent """
        if self.first_player(board):
            return PLAYER_X
        else:
            return PLAYER_O

    def winning(self, board, player):
        """ Checks if the player is winning - and which position is the
        winning move """
        board = board
        board_t = board.transpose()
        board_ = np.fliplr(board)

        pos = EMPTY
        # Rows and Columns
        for row in xrange(len(board)):
            if board[row].sum() == 2 * player:
                col = list(board[row]).index(0)
                pos = (row, col)
        for col in xrange(len(board_t)):
            if board_t[col].sum() == 2 * player: 
                row = list(board_t[col]).index(0)
                pos = (row, col)
        # Diagonals
        if board.trace() == 2 * player:
            i = list(board.diagonal()).index(0)
            pos = (i, i)
        elif board_.trace() == 2 * player:
            i = list(board_.diagonal()).index(0)
            pos = (i, 2-i)

        return pos

    def first_player_strategy(self, board, actions):
        """Strategy for the first player"""
        move_count = self.count_moves(board)
        if move_count == 0:
            # Play any corner
            action = choose(free_corners(actions))
        elif move_count == 1:
            # If opponent takes the center
            if has_center(board, PLAYER_O):
                # Play the opposite corner as before
                corner = get_corners(board, PLAYER_X)[0]
                return flip(corner)
            # Otherwise take the center
            else:
            # If opponent moved to a corner
                action = CENTER
        elif move_count == 2:
            # Take a corner move
            action = choose(free_corners(actions))
        else:
            action = choose(actions)
        return action

    def second_player_strategy(self, board, actions):
        """Strategy for the second player"""
        # If the opponent has played the center, just play corners
        if has_center(board, PLAYER_X):
            if len(free_corners(actions)) > 0:
                action = choose(free_corners(actions))
            else:
                action = choose(actions)
        else:
            move_count = self.count_moves(board)
            if move_count == 0:
                # The opponent has not played center, play center
                action = CENTER
            elif move_count == 1:
                has_edge_X = has_edge(board, PLAYER_X)
                has_corner_X = has_corner(board, PLAYER_X)
                # If the opponent has played a corner and an edge
                if has_edge_X and has_corner_X:
                    # Play the opposite corner
                    corner = get_corners(board, PLAYER_X)[0]
                    return flip(corner)
                elif has_edge_X and not has_corner_X:
                    edges = get_edges(board, PLAYER_X)
                    # Take the corner bordered by any of the edges
                    shifts = [(-1, 0), (0, -1), (0, 1), (1, 0)]

                    corners0 = [ tuple_add(edges[0], m) for m in shifts]
                    corners0 = [ m for m in corners0 if m in CORNERS ]

                    corners1 = [ tuple_add(edges[1], m) for m in shifts]
                    corners1 = [ m for m in corners1 if m in CORNERS ]

                    common_corner = list(set(corners0).intersection(corners1))
                    # If edges border a corner, take it
                    if len(common_corner) > 0:
                        action = common_corner[0]
                    else:
                        corners = list(set(corners0).union(corners1))
                        action = choose(corners)
                elif not has_edge(board, PLAYER_X) and has_corner(board, PLAYER_X):
                    action = choose(free_edges(actions))
            else:
                action = choose(actions)
        return action

    def act(self, state, actions, reward, episode_ended):
        # Automatic behaviour - if winning
        pos = self.winning(state, self.mark(state))
        if pos != EMPTY:
            return pos
        # Automatic behaviour - if threatened
        pos = self.winning(state, -1*self.mark(state))
        if pos != EMPTY:
            return pos

        # Strategy behaviour
        if self.first_player(state):
            return self.first_player_strategy(state, actions)
        else:
            return self.second_player_strategy(state, actions)

