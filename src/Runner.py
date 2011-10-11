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

def run(env, agent, epochs):
    """ Simulate some epochs of running """

    state = env.start()
    reward = 0
    episode_ended = True
    ret = []
    decision_table = {}

    epoch = 0
    episode_epochs = decisions = 0
    started = False # Only collect after a while
    while epoch < epochs:
        action = agent.act(state, reward, episode_ended)
        decisions += 1
        state, reward, episode_ended = env.react(action)

        # Add rewards to ret
        if isinstance( action, Option ):
            # If this was an option, then multiple rewards would have been
            # returned.
            ret += reward
            epoch += len( state ) - 1
            episode_epochs += len( state ) - 1
        else:
            ret.append( reward )
            epoch += 1
            episode_epochs += 1

        if not started and epoch > int( epochs * 0.9 ):
            started = True
            decision_table = {}
            episode_epochs = decisions = 0

        if started and episode_ended:
            decisions_, count = decision_table.get( episode_epochs, (0,0) )
            decision_table[ episode_epochs ] = decisions_ + decisions, count+1
            episode_epochs = decisions = 0

    # Chop off any extras
    return ret[ : epochs ], decision_table

