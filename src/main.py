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
import Runner
from ProgressBar import ProgressBar

def main( iterations, ensembles, epochs, agent_type, agent_args, env_type, env_args, file_prefix ):
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

    progress = ProgressBar( 0, ensembles*iterations, mode='fixed' )
    # Needed to prevent glitches
    oldprog = str(progress)

    # Counters
    ret = np.zeros( epochs, dtype=float )
    min_, max_ = np.inf * np.ones( epochs, dtype=float) , -np.inf * np.ones( epochs, dtype=float)
    var = np.zeros( epochs, dtype=float )

    env = env_type.create( *env_args )
    for i in xrange( 1, iterations+1 ):
        env = env.domain.reset_rewards( env, *env_args )

        ret_ = np.zeros( epochs, dtype=float )
        # Initialise environment and agent
        for j in xrange( 1, ensembles+1 ):
            agent = agent_type( env.Q, *agent_args )
            ret__ = Runner.run( env, agent, epochs )
            ret__ = np.cumsum( ret__ )
            # Add to ret_
            ret_ += (ret__ - ret_) / j

            # print progress
            progress.increment_amount()
            if oldprog != str(progress):
                print progress, "\r",
                sys.stdout.flush()
                oldprog=str(progress)

        ret += (ret_ - ret) / i
        min_ = np.min( np.vstack( ( min_, ret_ ) ), axis=0 )
        max_ = np.max( np.vstack( ( max_, ret_ ) ), axis=0 )

        var_ = np.power( ret_, 2 )
        var += (var_ - var) / i
    print "\n"

    var = np.sqrt( var - np.power( ret, 2 ) )

    f = open("%s-return.dat"%( file_prefix ), "w")
    # Print ret
    for i in xrange( len( ret ) ):
        f.write( "%d %f %f %f %f\n"%( i+1, ret[ i ], min_[i], max_[i], var[ i ] ) )
    f.close()

def print_help(args):
    """Print help"""
    print "Usage: %s <episodes> <epochs> <agent:args> <environment:args>" % (args[0])

def convert(arg):
    """Convert string arguments to numbers if possible"""
    if arg.isdigit():
        return int(arg)
    elif re.match("[0-9]*\.[0-9]+", arg):
        return float(arg)
    elif re.match("[0-9]*e[0-9]+", arg):
        return int(float(arg))
    else:
        return arg

if __name__ == "__main__":
    import sys
    def main_wrapper():
        """Wrapper around the main call - converts input arguments"""
        if "-h" in sys.argv[1:]:
            print_help(sys.argv)
            sys.exit( 0 )
        elif len(sys.argv) <> 7:
            print "Invalid number of arguments"
            print_help(sys.argv)
            sys.exit( 1 )
        else:
            iterations = convert( sys.argv[1] )
            ensembles = convert( sys.argv[2] )
            epochs = convert( sys.argv[3] )

            agent_str = sys.argv[4].split(":")
            agent_args = map( convert, agent_str[1:] )
            agent_type = Runner.load_agent( agent_str[0] )

            env_str = sys.argv[5].split(":")
            env_args = map( convert, env_str[1:] )
            env_type = Runner.load_env( env_str[0] )

            file_prefix = sys.argv[ 6 ]

            main( iterations, ensembles, epochs, agent_type, agent_args, env_type, env_args, file_prefix )

    main_wrapper()

