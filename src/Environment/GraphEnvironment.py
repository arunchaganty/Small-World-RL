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
        self.graph, self.rewards, self.reward_bias, self.end_states = self.generate_graph()
        self.size = self.graph.shape[0]
        self.pos = None

    def generate_graph(self):
        """Generates the defining graph"""
        raise NotImplemented()

    def _start(self):
        """Initialise the Environment
        @returns initial state and valid actions
        """
        # Choose a random node in the graph
        node = np.random.randint( self.size )
        self.pos = node
        return node, self.graph[ node, : ].nonzero()[1]

    def _react(self, action):
        """React to action
        @returns new state and valid actions, and reward, and if episode has
        ended
        """
        if self.graph[ self.pos, action ] == 0:
            raise ValueError( "%s -> %s not a valid action"%(self.pos,action,) )

        node = action
        self.pos = node
        actions = self.graph[ node, : ].nonzero()[1]
        reward = self.rewards[ 0, node ] + self.reward_bias
        episode_ended = bool( self.end_states[ 0, node ]  )

        if episode_ended:
            node, actions = self._start()
            
        return node, actions, reward, episode_ended
    
    def to_dot(self):
        graph_size = self.graph.shape[0]

        s = ""
        # Print header
        s += "graph {\n"
        # For every vertex, print a node 
        for i in xrange( graph_size ):
            s += '%d [label=""];\n'%(i)
        # For every edge
        for i,j in zip( *self.graph.nonzero() ):
            if i < j:
                s += '%d -- %d [label=""];\n'%(i,j)
        s += "}\n"

        return s

