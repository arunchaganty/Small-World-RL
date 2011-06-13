#!/usr/bin/env python
"""
RL TestBed
"""

import Runner

import re
import collections
import Agent
import Environment

class ArgumentError(StandardError):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return self.msg

def print_status( percent ):
    sys.stderr.write( "\rProgress: %f%%"%(percent) )

def main(epochs, agent_str, agent_args, env_str, env_args, verbose):
    """RL Testbed.
    @arg epochs: Number of episodes to run for
    @arg agent_str: String name of agent
    @arg agent_args: Arguments to the agent constructor
    @arg env_str: String name of environment
    @arg env_args: Arguments to the environment constructor
    """
    # Load agent and environment

    # Initialise environment and agent
    agent, env = load(agent_str, agent_args, env_str, env_args)
    runner = Runner.Runner(agent, env)
    total_rewards = runner.run(epochs)
    f = open( "rl.out", "w")
    i = 0
    for r in total_rewards:
        f.write( "%d %f\n"%(i, r) )
        i+=1
    f.close()


def load(agent_str, agent_args, env_str, env_args):
    """Try to load a class for agents or environment"""
    try:
        mod = __import__("Agent.%s"%(agent_str), fromlist=[Agent])
        assert( hasattr(mod, agent_str) )
        agent = getattr(mod, agent_str)
        agent = agent(*agent_args)
    except (ImportError, AssertionError):
        raise ArgumentError("Agent '%s' could not be found"%(agent_str))
    except (TypeError):
        raise ArgumentError("Agent '%s' could not be initialised\n%s"%(agent_str,agent.__init__.__doc__))

    try:
        mod = __import__("Environment.%s"%(env_str), fromlist=[Environment])
        assert( hasattr(mod, env_str) )
        env = getattr(mod, env_str)
        env = env(*env_args)
    except (ImportError, AssertionError):
        raise ArgumentError("Environment '%s' could not be found"%(env_str))
    except (TypeError):
        raise ArgumentError("Environment '%s' could not be initialised\n%s"%(env_str,env.__init__.__doc__))

    return agent, env

def print_help(args):
    """Print help"""
    print "Usage: %s [-v] <epochs> <agent> <environment>" % (args[0])

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
        try:
            if "-h" in sys.argv[1:]:
                print_help(sys.argv)
            elif len(sys.argv) < 4:
                raise ArgumentError("Too few arguments")
            elif len(sys.argv) > 5:
                raise ArgumentError("Too many arguments")
            else:
                verbose = False
                if sys.argv[1] == "-v":
                    verbose = True
                    sys.argv = sys.argv[0:1] + sys.argv[2:]

                epochs = sys.argv[1]
                if not epochs.isdigit():
                    raise ArgumentError("Epochs must be a valid integer")
                else:
                    epochs = int(sys.argv[1])

                agent_str = sys.argv[2].split(":")
                agent_args = map( convert, agent_str[1:] )
                agent_str = agent_str[0]

                env_str = sys.argv[3].split(":")
                env_args = map( convert, env_str[1:] )
                env_str = env_str[0]

                main(epochs, agent_str, agent_args, env_str, env_args, verbose)
        except ArgumentError as error:
            print "[Error]: %s" % (str(error))
            print_help(sys.argv)
            sys.exit(1)
    main_wrapper()
