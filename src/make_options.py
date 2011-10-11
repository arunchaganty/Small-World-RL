#!/usr/bin/env python
"""
RL Framework
Authors: Arun Chaganty
Make small world options

Take an environment, and an agent. Simulate the agent on various
reward instances of the environment, and use the generated policies to
construct options.
"""

import re
import numpy as np
import pickle

from Agent import *
from Environment import *
import OptionGenerator
import Runner

def main( epoch_budget, count, gen_type, gen_args, agent_type, agent_args, env_type, env_args, file_prefix ):
    """
    @arg epochs: Maximum number of epochs to use
    @arg count: Number of options to learn
    @arg agent_type: String name of agent
    @arg agent_args: Arguments to the agent constructor
    @arg env_type: String name of environment
    @arg env_args: Arguments to the environment constructor
    """

    env = env_type.create( *env_args )

    if gen_type == "betweenness":
        options = OptionGenerator.learn_options_from_betweenness( epoch_budget, count, env, agent_type, agent_args )
    elif gen_type == "small-world":
        options = OptionGenerator.learn_options_from_small_world( epoch_budget, count, env, agent_type, agent_args, *gen_args )
    else:
        raise NotImplemented()

    # Save options
    f = open("%s.options"%( file_prefix ), "w")
    pickle.dump( options, f )
    f.close()

def print_help(args):
    """Print help"""
    print "Usage: %s <epoch_budget> <count> <gen:args> <agent:args> <environment:args>" % (args[0])

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
            epoch_budget = convert( sys.argv[1] )
            count = convert( sys.argv[2] )

            gen_str = sys.argv[3].split(":")
            gen_args = map( convert, gen_str[1:] )
            gen_type = gen_str[0]

            agent_str = sys.argv[4].split(":")
            agent_args = map( convert, agent_str[1:] )
            agent_type = Runner.load_agent( agent_str[0] )

            env_str = sys.argv[5].split(":")
            env_args = map( convert, env_str[1:] )
            env_type = Runner.load_env( env_str[0] )

            file_prefix = sys.argv[ 6 ]

            main( epoch_budget, count, gen_type, gen_args, agent_type, agent_args, env_type, env_args, file_prefix )

    main_wrapper()

