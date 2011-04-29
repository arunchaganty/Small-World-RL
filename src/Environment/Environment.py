"""
Environment Base Class
"""

class Environment:
    state = None

    def __init__(self):
        pass

    def start(self):
        """Initialise the Environment
        @returns initial state and valid actions
        """
        return self._start()

    def _start(self):
        """Initialise the Environment
        @returns initial state and valid actions
        """
        raise NotImplemented()

    def react(self, action):
        return self._react( action )

    def _react(self, action):
        """React to action
        @returns new state and valid actions, and reward, and if episode has
        ended
        """
        raise NotImplemented()

    def optimal_actions(self, state, actions):
        """Return optimal action for state-action
        @returns optimal action(s)
        """
        raise NotImplemented()

