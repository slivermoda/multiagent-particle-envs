#!/usr/bin/env python
import os, sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse

from multiagent.environment import DynamicMultiAgentEnv
from multiagent.policy import InteractivePolicy
import multiagent.scenarios as scenarios

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='simple_rescue_and_trap.py',
                        help='Path of the scenario Python script.')
    args = parser.parse_args()

    # load scenario from script
    scenario = scenarios.load(args.scenario).Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = DynamicMultiAgentEnv(world,
                               scenario.update,
                               scenario.reset_world,
                               scenario.reward,
                               scenario.observation,
                               done_callback=scenario.done,
                               info_callback=None,
                               shared_viewer=False)
    # render call to create viewer window (necessary only for interactive policies)
    env.render()
    # create interactive policies for each agent
    policies = [InteractivePolicy(env, i) for i in range(env.n)]
    # execution loop
    obs_n = env.reset()
    while True:
        # query for action from each agent's policy
        act_n = []
        for i, policy in enumerate(policies):
            act_n.append(policy.action(obs_n[i]))
        # step environment
        obs_n, reward_n, done, _ = env.step(act_n)
        # render all agent views
        env.render()
        if done:
            world = scenario.make_world()
            env.reset(world)
        # display rewards
        # for agent in env.world.agents:
        #    print(agent.name + " reward: %0.3f" % env._get_reward(agent))
