"""
ArbitraryNavigationOptions Environment
"""

import numpy as np
import networkx as nx
import pdb

from Environment import *
import OptionGenerator
from ArbitraryNavigation import ArbitraryNavigation

class ArbitraryNavigationOptions( ):

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

        # Percentage
        if isinstance(count,str):
            count = int(count[:-1])
            count = count*env.S/100

        # Add options for all the optimal states
        O = []
        if scheme == "none":
            pass
        elif scheme == "random-node":
            O = OptionGenerator.optimal_options_from_random_nodes( env, count, *args )
        elif scheme == "random-path":
            O = OptionGenerator.optimal_options_from_random_paths( env, count, *args )
        elif scheme == "betweenness":
            O = OptionGenerator.optimal_options_from_betweenness( env, count, *args )
        elif scheme == "small-world":
            O = OptionGenerator.optimal_options_from_small_world( env, count, *args )
        elif scheme == "betweenness+small-world":
            O = OptionEnvironment.optimal_options_from_betweenness( env, count )
            count_ = count - len( O ) 
            O += OptionEnvironment.optimal_options_from_small_world( env, count_, *args )
        elif scheme == "load":
            O = OptionGenerator.options_from_file( *args )[:count]
        else:
            raise NotImplemented() 

        return OptionEnvironment( ArbitraryNavigationOptions, env.S, env.A, env.P, env.R, env.R_bias, env.start_set, env.end_set, O )

    @staticmethod
    def reset_rewards( env, road_map ):
        O = env.O
        env = ArbitraryNavigation.reset_rewards( env, road_map )
        return OptionEnvironment( ArbitraryNavigationOptions, env.S, env.A, env.P, env.R, env.R_bias, env.start_set, env.end_set, O )

