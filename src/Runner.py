"""
RL Framework
Authors: Arun Chaganty
Responsible for running agents and interacting with the Environment
"""

from Agent import *
from Environment import *
import Agents
import Environments

def load_env( env_type ):
    """Try to construct an environment"""
    mod = __import__("Environments.%s"%(env_type), fromlist=[Environments])
    assert( hasattr(mod, env_type) )
    envClass = getattr( mod, env_type )
    return envClass

def load_agent( agent_type ):
    """Try to construct an agent"""

    mod = __import__("Agents.%s"%(agent_type), fromlist=[Agents])
    assert( hasattr(mod, agent_type) )
    agentClass = getattr( mod, agent_type )
    return agentClass

def run(env, agent, episodes):
    """ Simulate some episodes of running """

    state, reward, episode_ended = env.start(), 0, True

    episodic_return, episodic_epochs = [], []
    ret, epochs = 0, 0

    episode = 0
    while episode < episodes:
        action = agent.act(state, reward, episode_ended)
        state, reward, episode_ended = env.react(action)

        # Add rewards to ret
        if isinstance( action, Option ):
            # If this was an option, then multiple rewards would have been
            # returned.
            ret += sum( reward )
            epochs += len( state ) - 1
        else:
            ret += reward
            epochs += 1

        if episode_ended:
            episodic_return.append( ret )
            episodic_epochs.append( epochs )
            epochs = 0



    # Chop off any extras
    return ret[ : epochs ]

