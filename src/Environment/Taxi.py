"""
Taxi Environment
"""

import numpy as np
# import scipy
# import scipy.sparse as sparse
import networkx as nx
import GraphEnvironment

class Taxi(GraphEnvironment.GraphEnvironment):
    """
    Taxi Environment
    Expects specification (size, endpoints, barriers) to be given
    """

    # State represented by a 3-value 3x3 board (X, O, -)
    DEFAULT_SPEC = (5, 4, 2)
    road_map = None
    starts = None
    steps = 1
    best_steps = 1
    
    LEFT    = 2**1
    RIGHT   = 2**2
    UP      = 2**3
    DOWN    = 2**4
    PICKUP  = 2**5
    PUTDOWN = 2**6

    REWARD_BIAS = -1
    REWARD_FAILURE = -10 - REWARD_BIAS
    REWARD_SUCCESS = 50 - REWARD_BIAS
    REWARD_CHECKPOINT = 0.0 - REWARD_BIAS

    # Environment Interface
    def __init__(self, spec, max_steps=500 ):
        """
        @spec - Specification (size, endpoints, barriers); either exactly
                specified in a file, or with numeric values in a list
        @max_steps - Number of steps that need to taken
        """

        # Check if the spec is a file
        if spec and spec.find(',') == -1: 
            self.road_map, self.starts = self.__load_file( spec )
        else:
            if not spec:
                spec = DEFAULT_SPEC
            else:
                spec = map(int, spec.split(',') )
            if len( spec != 3 ):
                raise ArgumentError("Incomplete specification")
            self.road_map, self.starts = self.__generate( *spec )

        self.max_steps = max_steps

        GraphEnvironment.GraphEnvironment.__init__(self)

    def __str__(self):
        road_size = self.road_map.shape[0]
        starts = len(self.starts)
        coords = self.pos % (road_size ** 2)
        dest = (self.pos / (road_size ** 2)) % starts
        pasn = (self.pos / (road_size ** 2 * starts)) % starts

        dest_str = chr(ord('A') + dest)
        if pasn < dest: 
            pasn_str = chr(ord('A') + pasn)
        elif pasn < starts - 1:
            pasn_str = chr(ord('A') + pasn + 1)
        elif pasn == starts - 1:
            pasn_str = 'T'

        val = ""
        val += "[Taxi P:%s D:%s]\n"%(pasn_str, dest_str)
        for j in xrange( road_size ):
            for i in xrange( road_size ):
                tile = int( self.road_map[ j, i ] )
                pos = j*road_size  + i
                val += '|'
                if pos == coords:
                    val_ = "*"
                elif tile == 0:
                    val_ = " "
                elif tile / 2 > 0:
                    tile = tile / 2
                    val_ = chr(ord('A') - 1 + tile)
                elif tile % 2 == 1:
                    val_ = "]"
                val += val_
            val += '|\n'
        return val
    
    def __repr__(self):
        return "[Taxi %d]" % (id(self))

    # Get state index
    def get_state(self, passenger, dest, posx, posy ):
        in_taxi = starts = len(self.starts)
        road_size = self.road_map.shape[0]

        st, offset = posx, road_size
        st, offset = st + offset * posy, offset * road_size
        st, offset = st + offset * dest, offset * starts
        # Some logic to handle the special definition of the passenger =
        # destination state
        if passenger > dest or passenger == in_taxi:
            st, offset = st + offset * ( passenger - 1 ), offset * starts 
        else:
            st, offset = st + offset * passenger, offset * starts 
        # The last state is reserved for passenger = destination
        if passenger == dest: 
            st = offset
        return st

    def unget_state(self, state):
        in_taxi = starts = len(self.starts)
        road_size = self.road_map.shape[0]

        posx, state = state % road_size, state / road_size 
        posy, state = state % road_size, state / road_size 
        dest, state = state % starts, state / starts 
        pasn, state = state % starts, state / starts 
        return pasn, dest, posx, posy

    def _start( self ):
        # Find the optimality of the previous run
        optimality = float( self.steps ) / self.best_steps
        #print "Optimality: %d %d %f "%( self.steps, self.best_steps, optimality )
        print "%f "%( optimality )

        starts = len(self.starts)
        road_size = self.road_map.shape[0]

        state = np.random.randint( (starts-1) * starts * (road_size**2) )
        self.pos = state
        self.best_steps = self.__get_min_step_count( state )
        self.steps = 0

        return state, tuple( self.graph.neighbors( state ) )

    def _react( self, action ):
        self.steps += 1
        if self.steps > self.max_steps:
            node, actions = self._start()
            reward = self.REWARD_FAILURE
            episode_ended = True
            return node, actions, reward, episode_ended
        else:
            return GraphEnvironment.GraphEnvironment._react( self, action )

    def generate_graph(self):
        """Construct a state-space graph
        Graph created with nodes representing (passenger, destination, taxi_x, taxi_y);
        There is only one state for passenger = destination.
        """

        # State space vector: (locs x dests x (starts - dest + taxi))
        in_taxi = starts = len(self.starts)
        road_size = self.road_map.shape[0]
        size = (road_size**2) * (starts) * (starts) + 1
        graph = np.zeros( (size, size), dtype = np.int )
        get_state = self.get_state

        # Road map matrix (Symmetrise?)
        road_graph = np.zeros( (road_size**2, road_size**2), dtype = np.int )
        for j in xrange( road_size ):
            for i in xrange( road_size ):
                if i > 0 and self.road_map[j,i-1] % 2 <> 1:
                    road_graph[ road_size * (j) + (i), road_size * (j) + (i-1)] = self.LEFT
                if i < road_size-1 and self.road_map[j,i] % 2 <> 1: 
                    road_graph[ road_size * (j) + (i), road_size * (j) + (i+1)] = self.RIGHT
                if j > 0:
                    road_graph[ road_size * (j) + (i), road_size * (j-1) + (i)] = self.UP
                if j < road_size-1: 
                    road_graph[ road_size * (j) + (i), road_size * (j+1) + (i)] = self.DOWN

        # Patch together to get graph
        for dest in xrange(starts):
            # When passenger \notin Taxi
            for start in xrange(starts):
                if start != dest:
                    graph[ get_state(start, dest, 0, 0):get_state(start, dest, road_size-1, road_size-1)+1,
                        get_state(start, dest, 0, 0):get_state(start, dest, road_size-1, road_size-1)+1] = road_graph 
                    # When pos == start -> Taxi
                    graph[ get_state(start, dest, *self.starts[start]), get_state(in_taxi, dest, *self.starts[start]) ] |= self.PICKUP
            # When passenger \in Taxi
            graph[ get_state(in_taxi, dest, 0, 0):get_state(in_taxi, dest, road_size-1, road_size-1)+1, 
                get_state(in_taxi, dest, 0, 0):get_state(in_taxi, dest, road_size-1, road_size-1)+1] = road_graph 
            # When pos == dest => stop.
            graph[ get_state(in_taxi, dest, *self.starts[dest]), get_state(dest, dest, *self.starts[dest]) ] |= self.PUTDOWN
            # Set the reward states and end states

        # Use the above adj. matrix to create the below graph
        graph = nx.DiGraph( graph, reward_bias = self.REWARD_BIAS )

        # Set attributes
        for dest in xrange(starts):
            for start in xrange(starts):
                if start != dest:
                    graph.edge[ get_state(start, dest, *self.starts[start])][ 
                            get_state(in_taxi, dest, *self.starts[start]) ]["reward"] = self.REWARD_CHECKPOINT
            graph.node[ get_state(dest, dest, *self.starts[dest]) ]["reward"] = self.REWARD_SUCCESS
            graph.node[ get_state(dest, dest, *self.starts[dest]) ]["end?"] = True

        return graph

    def __load_file( self, spec_file ):
        """
        Load a specification file:
        <grid-size>
        <start#> (<x>,<y>) (<x>,<y>) ...
        <edge#> (<x>,<y>,<u>,<l>) (<x>,<y>,<u>,<l>) ...
        """

        spec = map( str.strip, open( spec_file ).readlines() )
        size = int( spec[0] )
        road_map = np.zeros( (size,size), dtype = np.int )

        # HACK: Unsafe code here
        starts = map(eval, spec[1].split())
        edges = map(eval, spec[2].split())

        for x0, y0, up, length in edges:
            for i in xrange(length):
                road_map[ y0 + ((up^1)*i), x0 + up*i ] = 1

        for i in xrange(len(starts)):
            tuple( reversed(starts[i]) )
            road_map[ starts[i] ] += 2*(i+1)

        return road_map, starts

    def __generate( self, size, starts, edges ):
        """
        Generate a random map
        @size - size of map
        @starts - No. of starts
        @edges - No. of edges
        """

        road_map = np.zeros( (size, size), dtype = np.int )

        starts = [ (np.random.randint(size), np.random.randint(size)) ] * len( starts )
        # TODO: Something that makes sense here
        edges = [ (np.random.randint(size), np.random.randint(size), 0, 2) ] * len( edges )

        for i in xrange(len(starts)):
            road_map[ starts[i] ] = -i
        for x0, y0, up, length in edges:
            for i in xrange(length):
                road_map[ y0 + ((up^1)*i), x0 + up*i ] = 1
        return road_map, starts

    def __get_min_step_count( self, state ):
        goal = self.get_state( 0, 0, 0, 0 ) # Just a short cut to get goal state value
        return nx.shortest_path_length( self.graph, state, goal )

