#!/usr/bin/env python
"""
RL Framework
Authors: Arun Chaganty
Entry point
"""

import re
import numpy as np

from Agent import *
from Environment import *
from Runner import Runner

def main( iterations, ensembles, epochs, agent_type, agent_args, env_type, env_args ):
    """RL Testbed.
    @arg iterations: Number of environments to average over
    @arg ensembles: Number of bots to average over
    @arg epochs: Number of episodes to run for
    @arg agent_type: String name of agent
    @arg agent_args: Arguments to the agent constructor
    @arg env_type: String name of environment
    @arg env_args: Arguments to the environment constructor
    """
    # Load agent and environment

    ret = np.zeros( epochs, dtype=float )
    var = np.zeros( epochs, dtype=float )
    for i in xrange( 1, iterations+1 ):
        env = Runner.load_env( env_type, env_args )
        runner = Runner( env )

        # Print a graph of the environment
        # open( "graph-i%d.dot"%(i), "w" ).write( env.to_dot() )

        ret_ = np.zeros( epochs, dtype=float )
        var_ = np.zeros( epochs, dtype=float )
        # Initialise environment and agent
        for j in xrange( 1, ensembles+1 ):
            agent = Runner.load_agent( env, agent_type, agent_args ) 
            ret__ = runner.run( agent, epochs )
            ret__ = np.cumsum( ret__ )
            var__ = np.power( ret__, 2 )

            # Add to ret_
            ret_ += (ret__ - ret_) / j
            var_ += (var__ - var_) / j

        ret += (ret_ - ret) / i
        var += (var_ - var) / i

    var = np.sqrt( var - np.power( ret, 2 ) )
    # Print ret
    for i in xrange( len( ret ) ):
        print "%d %f %f"%( i+1, ret[ i ], var[ i ] )

    # Dump the policy learnt?

def print_help(args):
    """Print help"""
    print "Usage: %s <episodes> <epochs> <agent:args> <environment:args>" % (args[0])

def convert(arg):
    """Convert string arguments to numbers if possible"""
    if arg.isdigit():
        return int(arg)
    elif re.match("[0-9]*\.[0-9]+", arg):
        return float(arg)
    else:
        return arg

if __name__ == "__main__":
    import sys
    def main_wrapper():
        """Wrapper around the main call - converts input arguments"""
        if "-h" in sys.argv[1:]:
            print_help(sys.argv)
            sys.exit( 0 )
        elif len(sys.argv) <> 6:
            print "Invalid number of arguments"
            print_help(sys.argv)
            sys.exit( 1 )
        else:
            iterations = convert( sys.argv[1] )
            ensembles = convert( sys.argv[2] )
            epochs = convert( sys.argv[3] )

            agent_str = sys.argv[4].split(":")
            agent_args = map( convert, agent_str[1:] )
            agent_type = agent_str[0]

            env_str = sys.argv[5].split(":")
            env_args = map( convert, env_str[1:] )
            env_type = env_str[0]

            main( iterations, ensembles, epochs, agent_type, agent_args, env_type, env_args )

    main_wrapper()

