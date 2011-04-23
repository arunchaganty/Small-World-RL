"""
TicTacToe Environment
"""

import numpy as np
import copy
import Environment
import Agent

PLAYER_X = -1
PLAYER_N = 0
PLAYER_O = 1

START_P = 1
START_O = 2
START_RND = 3

# Load an agent
def load(agent, agent_args):
    """Load an agent"""
    try:
        mod = __import__("Agent.%s"%(agent), fromlist=[Agent])
        assert( hasattr(mod, agent) )
        agent = getattr(mod, agent)
        agent = agent(*agent_args)
    except (ImportError, AssertionError):
        raise ValueError("Agent '%s' could not be found"%(agent))
    return agent

class TicTacToeAfterState(Environment.Environment):
    """
    TicTacToe Environment
    Expects starting position and opponent Agent to be given
    """

    opponent = None
    opponent_mark = None
    starting_policy = START_RND

    # State represented by a 3-value 3x3 board (X, O, -)
    board = None

    # Environment Interface
    def __init__(self, starting_policy, opponent, *opponent_args):
        """
        @starting_policy - Does the opponent start the game?
        @opponent - string name of the opponent
        @opponent_args - string args for the opponent
        """
        Environment.Environment.__init__(self)
        self.opponent = load(opponent, opponent_args)
        self.starting_policy = {
                "random":START_RND,
                "true":START_O,
                "True":START_O,
                "false":START_P,
                "False":START_P
                }[starting_policy]

        self.board = self.__init_board()

    def __str__(self):
        val = ""
        val += "[TicTacToe]\n"
        for row in self.board:
            for col in row:
                if col == PLAYER_X: 
                    val += 'X '
                elif col == PLAYER_N: 
                    val += '  '
                elif col == PLAYER_O: 
                    val += 'O '
            val = val[:-1] + '\n'
        return val
    
    def __repr__(self):
        return "[TicTacToe %d]" % (id(self))

    def start(self):
        board, actions, reward, episode_ended = self.restart(0)
        return board, actions

    def restart(self, reward):
        # Play a round with the opponent
        if self.starting_policy == START_O :
            opponent_starts = True
        elif self.starting_policy == START_RND and np.random.randint(2) == 1:
            opponent_starts = True
        else:
            opponent_starts = False

        if opponent_starts:
            self.opponent_mark = PLAYER_X
            board, actions = copy.deepcopy(self.board), self.__get_actions()
            action = self.opponent.act(board, actions, -1 * reward, True)
            # We can safely ignore the possibility that a game will end in 1
            # turn
            self.__apply_action(action, self.__opponent_mark())
        else:
            self.opponent_mark = PLAYER_O

        self.after_state = False

        return self.board, self.__get_actions(), reward, True

    def react(self, action):
        # Check action
        if not self.after_state:
            if action not in self.__get_actions():
                raise ValueError( "%s not a valid action"%(action,) )

            # Play player turn
            complete, reward = self.__apply_action(action, self.__player_mark())

            # Handle episode restart
            if complete:
                return self.restart(reward)

            self.after_state = True

            # Otherwise continue
            return self.board, [()], 0, False
        else:
            # Play opponent turn
            board, actions = copy.deepcopy(self.board), self.__get_actions()
            action = self.opponent.act(board, actions, 0, False)
            complete, reward = self.__apply_action(action, self.__opponent_mark())

            # Handle episode restart
            if complete:
                return self.restart(-1*reward)
            self.after_state = False

            # Otherwise continue
            return self.board, self.__get_actions(), reward, False

    def __init_board(self):
        """Return the empty board - all PLAYER_N"""
        return np.array([ [ PLAYER_N for i in xrange(3) ] for j in xrange(3) ])
    
    def __get_actions(self):
        """Get all valid actions for board state"""
        def get_actions_(board):
            """Enumerate valid actions"""
            for row in xrange(len(board)):
                for col in xrange(len(board[row])):
                    if self.board[row][col] == 0:
                        yield (row, col)
        return [ x for x in get_actions_(self.board) ]

    def __opponent_mark(self):
        """Get the mark of the opponent"""
        return self.opponent_mark

    def __player_mark(self):
        """Get the mark of the player"""
        if self.opponent_mark == PLAYER_X:
            return PLAYER_O
        else:
            return PLAYER_X

    def __apply_action(self, action, player):
        """Modify state, given the action
        @returns True if the game has restarted 
        """
        self.board[action[0]][action[1]] = player
        #print self

        # Check win
        winner = self.__check_winner()
        # If the winner function reports a winner, we are done.
        # If it returns PLAYER_N, we have to check if the board is complete,
        # before declaring the game done. This is done by 
        # self.__get_actions() == 0
        if winner != PLAYER_N or len(self.__get_actions()) == 0 :
            self.board = self.__init_board()
            if winner == self.__player_mark():
                reward = 2 
            elif winner == PLAYER_N:
                reward = 1
            else:
                reward = -1
            # reward = player * winner
            return True, reward
        else:
            return False, 0

    def __check_winner(self):
        """Find if there is a winner, and return it"""
        # Check all rows, columns and diagonals 
        board = self.board
        board_ = np.fliplr(board)

        # Rows and Columns
        for row, col in zip(board, board.transpose()):
            if row.sum() == 3 * PLAYER_X or col.sum() == 3*PLAYER_X:
                return PLAYER_X
            elif row.sum() == 3 * PLAYER_O or col.sum() == 3*PLAYER_O:
                return PLAYER_O

        # Diagonals
        if board.trace() == 3 * PLAYER_X or board_.trace() == 3 * PLAYER_X:
            return PLAYER_X
        elif board.trace() == 3 * PLAYER_O or board_.trace() == 3 * PLAYER_O:
            return PLAYER_O

        return PLAYER_N
   
