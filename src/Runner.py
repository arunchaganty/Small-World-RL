"""Responsible for running agents and interacting with the environment"""

import copy
import collections
import numpy as np
from Environment.OptionEnvironment import Option

class Runner:
    """Responsible for running agents and interacting with the environment"""
    def __init__(self, agent, env):
        """
        @arg Agent
        @arg env
        """
        self.agent = agent
        self.env = env

    def run(self, epochs):
        """ Simulate some epochs of running """

        state, actions = self.env.start()
        reward = 0
        episode_ended = True
        ret = []

        while epochs > 0:
            action = self.agent.act(state, actions, reward, episode_ended)
            #if self.post_act_hook: self.post_act_hook(self.env, self.agent, state, actions, action)
            state, actions, reward, episode_ended, epochs_run = self.env.react(action)
            #if self.post_react_hook: self.post_react_hook(self.env, self.agent, state, actions, reward, episode_ended)

            # Add rewards to ret
            if isinstance( action, Option ):
                # If this was an option, then multiple rewards would have been
                # returned.
                ret += reward
            else:
                ret.append( reward )

            epochs -= epochs_run

        return ret

