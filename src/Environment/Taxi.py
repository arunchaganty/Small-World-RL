"""
Taxi Environment
"""

import numpy as np
import scipy.sparse as sparse
import copy
import Environment
import GraphEnvironment
import Agent

# Load an agent
def load(agent, agent_args):
    """Load an agent"""
    try:
        mod = __import__("Agent.%s"%(agent), fromlist=[Agent])
        assert( hasattr(mod, agent) )
        agent = getattr(mod, agent)
        agent = agent(*agent_args)
    except (ImportError, AssertionError):
        raise ArgumentError("Agent '%s' could not be found"%(agent))
    return agent

class Taxi(GraphEnvironment.GraphEnvironment):
    """
    Taxi Environment
    Expects specification (size, endpoints, barriers) to be given
    """

    # State represented by a 3-value 3x3 board (X, O, -)
    DEFAULT_SPEC = (5,4,2)
    road_map = None
    starts = None

    LEFT    = 2**1
    RIGHT   = 2**2
    UP      = 2**3
    DOWN    = 2**4
    PICKUP  = 2**5
    PUTDOWN = 2**6

    # Environment Interface
    def __init__(self, spec=None):
        """
        @spec - Specification (size, endpoints, barriers); either exactly
                specified in a file, or with numeric values in a list
        """
        # Check if the spec is a file
        if spec and spec.find(',') == -1: 
            self.road_map, self.starts = self.load_file( spec )
        else:
            if not spec:
                spec = DEFAULT_SPEC
            else:
                spec = map(int, spec.split(',') )
            if len( spec != 3 ):
                raise ArgumentError("Incomplete specification")
            self.road_map, self.starts = self.generate( *spec )

        GraphEnvironment.GraphEnvironment.__init__(self)

    def __str__(self):
        val = ""
        val += "[Taxi]\n"
        for row in self.road_map:
            for col in row:
                if col == 0:
                    val_ = " "
                if col == 1:
                    val_ = "]"
                if col < 0:
                    val_ = "A" - col
                val += val_
            val += '\n'
        return val
    
    def __repr__(self):
        return "[Taxi %d]" % (id(self))

    @staticmethod
    def load_file( spec_file ):
        """
        Load a specification file:
        <grid-size>
        <start#> (<x>,<y>) (<x>,<y>) ...
        <edge#> (<x>,<y>,<u>,<l>) (<x>,<y>,<u>,<l>) ...
        """

        spec = map( str.strip, open( spec_file ).readlines() )
        size = int( spec[0] )
        road_map = np.zeros( (size,size) )

        # HACK: Unsafe code here
        starts = map(eval, spec[1].split())
        edges = map(eval, spec[2].split())

        for i in xrange(len(starts)):
            tuple( reversed(starts[i]) )
            road_map[ starts[i] ] = -i
        for x0,y0,d,l in edges:
            for i in xrange(l):
                road_map[ y0 + ((d^1)*i), x0 + d*i ] = 1

        return road_map, starts

    @staticmethod
    def generate( size, starts, edges ):
        """
        Generate a random map
        @size - size of map
        @starts - No. of starts
        @edges - No. of edges
        """

        road_map = np.zeros( size, size )

        starts = [ (np.random.randint(size), np.random.randint(size)) for i in starts ] 
        # TODO: Something that makes sense here
        edges = [ (np.random.randint(size), np.random.randint(size), 0, 2) for i in edges ] 

        for i in xrange(len(starts)):
            road_map[ starts[i] ] = -i
        for x0,y0,d,l in edges:
            for i in xrange(l):
                road_map[ x0 + d*l, y0 + (d^1)*l ] = 1
        return road_map, starts

    def generate_graph(self):
        """Construct a graph"""
        # State space vector: (locs x (starts + taxi) x starts)
        road_size = self.road_map.shape[0]
        size = (road_size**2) * (starts+1) * (starts)
        graph = sparse.lil_matrix( (size, size) )

        def get_state( dest, passenger, posx, posy ):
            st, offset = posy, road_size
            st, offset = st + offset * posx, offset * road_size
            st, offset = st + offset * passenger, offset + offset * (starts + 1)
            st, offset = st + offset * dest, offset + offset * starts
            assert(size == offset)
            return st

        # Road map matrix (Symmetrise?)
        road_graph = sparse.lil_matrix( self.road_size )
        for j in xrange( road_size ):
            for i in xrange( road_size ):
                if i != 0:
                    road_graph[ get_state(0,0,i,j), get_state(0,0,i-1,j)] = LEFT
                if i != road_size-1:
                    road_graph[ get_state(0,0,i,j), get_state(0,0,i+1,j)] = RIGHT
                if j != 0:
                    road_graph[ get_state(0,0,i,j), get_state(0,0,i,j-1)] = UP
                if j != road_size-1:
                    road_graph[ get_state(0,0,i,j), get_state(0,0,i,j+1)] = DOWN
        print road_graph.todense()

        return road_map

