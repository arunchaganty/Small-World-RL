"""
Graphable Base Class
"""

import Environment
import numpy as np

class GraphEnvironment(Environment.Environment):
    """Environment that defines a graph structure"""
    graph = None

    def __init__(self):
        Environment.Environment.__init__(self)
        self.graph = self.generate_graph()
        self.size = len( self.graph )
        self.pos = None

        self.generate_graph()

    def generate_graph(self):
        """Generates the defining graph"""
        raise NotImplemented()

    def start(self):
        """Initialise the Environment
        @returns initial state and valid actions
        """
        # Choose a random node in the graph
        node = np.random.randint( self.size )
        self.pos = node
        return node, graph[ node ] 

    def restart(self, reward):
        """Restarts the episode
        @returns new state and valid actions, and reward"""

        node = np.random.randint( self.size )
        self.pos = node
        return node, graph[ node ] 

    def react(self, action):
        """React to action
        @returns new state and valid actions, and reward, and if episode has
        ended
        """
        if self.graph[ self.pos, action ] == 0:
            raise ValueError( "%s -> %s not a valid action"%(self.pos,action,) )

        node = action
        self.pos = node
        return node, graph[ node ] 

