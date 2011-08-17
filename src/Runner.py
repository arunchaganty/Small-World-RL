"""
RL Framework
Authors: Arun Chaganty
Responsible for running agents and interacting with the Environment
"""

from Agent import *
from Environment import *
import Agents
import Environments

import pdb

class Runner:
    """Responsible for running agents and interacting with the environment"""
    def __init__(self, env):
        """
        @arg Agent
        @arg env
        """
        self.env = env

    @staticmethod
    def load_env( env_type, env_args ):
        """Try to construct an environment"""
        mod = __import__("Environments.%s"%(env_type), fromlist=[Environments])
        assert( hasattr(mod, env_type) )
        env = getattr( mod, env_type )
        env = env.create( *env_args )
        # except (ImportError, AssertionError):
        #     raise ValueError("Environment '%s' could not be found"%(env_type))
        # except (TypeError):
        #     raise ValueError("Environment '%s' could not be initialised\n%s"%(env_type, env.create.__doc__))

        return env

    @staticmethod
    def load_agent( env, agent_type, agent_args ):
        """Try to construct an agent"""

        mod = __import__("Agents.%s"%(agent_type), fromlist=[Agents])
        assert( hasattr(mod, agent_type) )
        agent = getattr( mod, agent_type )
        agent = agent( env.Q, *agent_args )
        # except (ImportError, AssertionError):
        #     raise ValueError("Agent '%s' could not be found"%(agent_type))
        # except (TypeError):
        #     raise ValueError("Agent '%s' could not be initialised\n%s"%(agent_type, agent.__init__.__doc__))

        return agent

    def run(self, agent, epochs):
        """ Simulate some epochs of running """

        state = self.env.start()
        reward = 0
        episode_ended = True
        ret = []

        epoch = 0
        while epoch < epochs:
            action = agent.act(state, reward, episode_ended)
            state, reward, episode_ended = self.env.react(action)

            # Add rewards to ret
            if isinstance( action, Option ):
                # If this was an option, then multiple rewards would have been
                # returned.
                ret += reward
                epoch += len( state ) - 1
            else:
                ret.append( reward )
                epoch += 1

        # Chop off any extras
        return ret[ : epochs ]

