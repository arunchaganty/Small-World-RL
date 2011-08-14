"""
Graphable Base Class
"""

import Environment
import numpy as np
import networkx as nx

class GraphEnvironment(Environment.Environment):
    """Environment that defines a graph structure"""
    graph = None

    def __init__(self):
        Environment.Environment.__init__(self)
        self.graph = self.generate_graph()
        if not self.graph.graph.has_key( "reward_bias" ):
            self.graph.graph["reward_bias"] = 0
        self.pos = None

    def generate_graph(self):
        """Generates the defining graph
        @return: Must be a networkx graph; appropriately has rewards on edges and nodes"""
        raise NotImplemented()

    def _start(self):
        """Initialise the Environment
        @returns initial state and valid actions
        """
        # Choose a random node in the graph
        node = np.random.randint( len( self.graph ) )
        self.pos = node
        return state, tuple( self.graph.neighbors( state ) )

    def _react(self, action):
        """React to action
        @returns new state and valid actions, and reward, and if episode has
        ended
        """
        if not self.graph.has_edge( self.pos, action ):
            raise ValueError( "%s -> %s not a valid action"%(self.pos,action,) )

        # Edge Reward
        reward = self.graph.edge[self.pos][action].get( "reward", 0 )

        self.pos = node = action
        actions = tuple( self.graph.neighbors( node ) )
        reward += self.graph.node[node].get( "reward", 0 )
        episode_ended = self.graph.node[node].get( "end?", False )

        bias = self.graph.graph["reward_bias"]
        if episode_ended:
            node, actions = self._start()
            
        return node, actions, reward + bias, episode_ended, 1
    
    def to_dot(self):
        graph_size = len(self.graph)

        s = ""
        # Print header
        s += "graph {\n"
        # For every vertex, print a node 
        for i in xrange( graph_size ):
            s += '%d [label=""];\n'%(i)
        # For every edge
        for i,j in self.graph.edges():
            if i < j:
                s += '%d -- %d [label=""];\n'%(i,j)
        s += "}\n"

        return s

