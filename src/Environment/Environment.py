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
        raise NotImplemented()

    def restart(self, reward):
        """Restarts the episode
        @returns new state and valid actions, and reward"""
        raise NotImplemented()

    def react(self, action):
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



