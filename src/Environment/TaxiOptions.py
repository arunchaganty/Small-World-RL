"""
TaxiOptions Environment
"""

import numpy as np
import scipy
import scipy.sparse as sparse
import OptionEnvironment
import Taxi
import networkx as nx
import pdb

class TaxiOptions(Taxi.Taxi, OptionEnvironment.OptionEnvironment):
    # Environment Interface
    def __init__(self, spec, option_scheme='none', max_steps=500 ):
        """
        @spec - Specification (size, endpoints, barriers); either exactly
                specified in a file, or with numeric values in a list
        @option_scheme - optimal|random|betweeness
        @max_steps - Number of steps that need to taken
        """

        Taxi.Taxi.__init__( self, spec, max_steps )
        # Add options for all the optimal states
        if option_scheme == "none":
            self.set_options( [] )
        elif option_scheme == "optimal":
            self.set_options( self.__get_optimal_options() )
        elif option_scheme == "random":
            self.set_options( self.__get_random_options() )
        else:
            raise NotImplemented() 

    def start(self):
        return OptionEnvironment.OptionEnvironment.start( self )

    def react(self, action):
        return OptionEnvironment.OptionEnvironment.react( self, action )
    
    def __make_option_from_paths( self, dest, paths ):
        paths.pop( dest )
        start = set( paths.keys() )
        stop = set( [dest] )
        policy = dict( [ (k,v[-2]) for k,v in paths.items() ] )

        option = OptionEnvironment.DeterministicOption( start, stop, policy )

        return option

    def __get_optimal_options( self ):
        # Reverse the graph to get the shortest paths to state
        gr = self.graph.reverse()
        options = []
        
        in_taxi = len( self.starts )
        road_size = len( self.road_map )
        for dest_i in xrange( len( self.starts ) ):
            # Paths to the passenger
            for start_i in xrange( len( self.starts ) ):
                if start_i == dest_i: 
                    continue
                st = self.get_state( start_i, dest_i, *self.starts[ start_i ] )
                paths = nx.shortest_path( gr, source=st )
                options.append( self.__make_option_from_paths( st, paths ) )
            # Paths to the the destination
            grs = gr.subgraph( range( 
                self.get_state( in_taxi, dest_i, 0, 0 ),
                self.get_state( in_taxi, dest_i, road_size, road_size-1 ) ) )
            st = self.get_state( in_taxi, dest_i, *self.starts[ dest_i ] )
            paths = nx.shortest_path( grs, source=st )
            options.append( self.__make_option_from_paths( st, paths ) )

        return options

    def __make_option_from_path( self, start, stop, path ):
        start = set( [start] )
        stop = set( [stop] )
        policy = dict( zip( path[:-1], path[1:] ) )

        option = OptionEnvironment.DeterministicOption( start, stop, policy )

        return option


    def __get_random_options( self, r = 2 ):
        # Get all the edges in the graph
        path_lengths = nx.shortest_path_length( self.graph )
        paths = nx.shortest_path( self.graph )

        options = []

        for node, node_dists in path_lengths.items():
            node_dists.pop( node )
            if not node_dists: 
                continue
            neighbours, dists = zip( *node_dists.items() )
            # Create a pr distribution
            dists = np.power( np.array( dists, dtype=float ), -r )
            # Zero out neighbours
            for i in xrange( len( dists ) ):
                if dists[i] == 1: dists[i] = 0
            if not dists.any(): 
                continue
            dists = dists / sum(dists)
            idx = np.random.multinomial(1, dists).argmax()

            dest = neighbours[ idx ]
            options.append( self.__make_option_from_path( node, dest, paths[node][dest] ) )

        return options

