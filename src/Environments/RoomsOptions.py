"""
RoomsOptions Environment
"""

import numpy as np
import networkx as nx
import pdb

from Environment import *
from Rooms import Rooms

class RoomsOptions( OptionEnvironment ):

    @staticmethod
    def create( spec, scheme = 'none', count = 20, *args ):
        """
        @spec - Specification (size, endpoints, barriers); either exactly
                specified in a file, or with numeric values in a list
        @option_scheme - none|manual|optimal|small-world|random|ozgur's betweenness|ozgur's randomness|end
        @n_actions - Number of steps that need to taken
        comment : optimal(shortest path to destination)??|random|ozgur's betweenness|ozgur's randomness
        """

        env = Rooms.create( spec )
        g = env.to_graph()
        gr = g.reverse()

        # Add options for all the optimal states
        O = []
        if scheme == "none":
            pass
        elif scheme == "manual":
            raise NotImplemented()
        elif scheme == "optimal":
            raise NotImplemented()
        elif scheme == "random-node":
            O = OptionEnvironment.make_options_from_random_nodes( g, gr, count, *args )
        elif scheme == "random-path":
            O = OptionEnvironment.make_options_from_random_paths( g, gr, count, False, *args )
        elif scheme == "mrandom-path":
            O = OptionEnvironment.make_options_from_random_paths( g, gr, count, True, *args )
        elif scheme == "betweenness":
            O = OptionEnvironment.make_options_from_betweenness( g, gr, count, *args )
        elif scheme == "small-world":
            O = OptionEnvironment.make_options_from_small_world( g, gr, count, False, *args )
        elif scheme == "msmall-world":
            O = OptionEnvironment.make_options_from_small_world( g, gr, count, True, *args )
        else:
            raise NotImplemented() 

        return OptionEnvironment( env.S, env.A, env.P, env.R, env.R_bias, env.start_set, env.end_set, O )

#    def __get_optimal_options( self ):
#        # Reverse the graph to get the shortest paths to state
#        # there is something natural options that sir was talking about
#        # here, I do not remember it well, but it was do the 
#        # options that are returned resemeble the actions that
#        # the agent can take in real world, and not some shady
#        # actions which we derive just based on the interaction 
#        # graph
#        # that is the sequence which these options represent can 
#        # actually be performed in real world
#        gr = self.graph.reverse()
#        options = []
#        
#        in_taxi = len( self.starts )
#        road_size = len( self.road_map )
#        for dest_i in xrange( len( self.starts ) ):
#            # Paths to the passenger
#            for start_i in xrange( len( self.starts ) ):
#                if start_i == dest_i: 
#                    continue
#                st = self.get_state( start_i, dest_i, *self.starts[ start_i ] )
#                paths = nx.shortest_path( gr, source=st )
#                options.append( self.__make_option_from_paths( st, paths ) )
#            # Paths to the destination
#            grs = gr.subgraph( range( 
#                self.get_state( in_taxi, dest_i, 0, 0 ),
#                self.get_state( in_taxi, dest_i, road_size-1, road_size-1 ) + 1 ) )
#            st = self.get_state( in_taxi, dest_i, *self.starts[ dest_i ] )
#            paths = nx.shortest_path( grs, source=st )
#            options.append( self.__make_option_from_paths( st, paths ) )
#
#        return options
#
#    def __get_manual_options( self ):
#        """Use manually-defined options"""
#        # Reverse the graph to get the shortest paths to state
#        gr = self.graph.reverse()
#        options = []
#        
#        in_taxi = len( self.starts )
#        road_size = len( self.road_map )
#        for dest_i in xrange( len( self.starts ) ):
#            # Paths to the passenger
#            for start_i in xrange( len( self.starts ) ):
#                if start_i == dest_i: 
#                    continue
#                for start_j in xrange( len( self.starts ) ):
#                    if start_j == dest_i: 
#                        continue
#                    # Paths to some pad
#                    st = self.get_state( start_i, dest_i, *self.starts[ start_j ] )
#                    paths = nx.shortest_path( gr, source=st )
#                    options.append( self.__make_option_from_paths( st, paths ) )
#            # Paths to the destination
#            grs = gr.subgraph( range( 
#                self.get_state( in_taxi, dest_i, 0, 0 ),
#                self.get_state( in_taxi, dest_i, road_size, road_size-1 ) ) )
#            for dest_j in xrange( len( self.starts ) ):
#                st = self.get_state( in_taxi, dest_i, *self.starts[ dest_j ] )
#                paths = nx.shortest_path( grs, source=st )
#                options.append( self.__make_option_from_paths( st, paths ) )
#
#        return options

