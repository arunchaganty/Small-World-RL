"""
ArbitraryNavigationOptions Environment
"""

import numpy as np
import networkx as nx
import pdb

from Environment import *
from ArbitraryNavigation import ArbitraryNavigation

class ArbitraryNavigationOptions( OptionEnvironment ):

    @staticmethod
    def create( height, width, scheme = 'none', count = 20, *args ):
        """
        @spec - Specification (size, endpoints, barriers); either exactly
                specified in a file, or with numeric values in a list
        @option_scheme - none|manual|optimal|small-world|random|ozgur's betweenness|ozgur's randomness|end
        @n_actions - Number of steps that need to taken
        comment : optimal(shortest path to destination)??|random|ozgur's betweenness|ozgur's randomness
        """

        env = ArbitraryNavigation.create( height, width )
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

