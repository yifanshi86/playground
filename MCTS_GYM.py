#!/usr/bin/env python2
import os
from gym.wrappers import monitor
import gym
import sys
import random
import itertools
from time import time
from copy import copy
from math import sqrt, log
import pommerman
from pommerman import agents


def moving_average(v, n):
    n = min(len(v), n)
    ret = [.0]*(len(v)-n+1)
    ret[0] = float(sum(v[:n]))/n
    for i in range(len(v)-n):
        ret[i+1] = ret[i] + float(v[n+i] - v[i])/n
    return ret


def ucb(node):
    return node.value / node.visits + sqrt(log(node.parent.visits)/node.visits)


def combinations(space):
    if isinstance(space, gym.spaces.Discrete):
        return range(space.n)
    elif isinstance(space, gym.spaces.Tuple):
        return itertools.product(*[combinations(s) for s in space.spaces])
    else:
        raise NotImplementedError


class Node:
    def __init__(self, parent, action):
        self.parent = parent
        self.action = action
        self.children = []
        self.explored_children = 0
        self.visits = 0
        self.value = 0


class Runner:
    #loops: number of episode to run
    #max_depth: the mcts tree depth.
    #playouts: for each episode, the maximum number of steps to run.
    def __init__(self, loops, env_name, max_depth=1000, playouts=10000):
        self.env_name = env_name
        self.loops = loops
        self.max_depth = max_depth
        self.playouts = playouts

    def print_stats(self, loop, score, avg_time):
        #print('\r%3d   score:%10.3f   avg_time:%4.1f s' % (loop, score, avg_time))
        print(loop, score, avg_time)
        #sys.stdout.write('\r%3d   score:%10.3f   avg_time:%4.1f s' % (loop, score, avg_time))
        #sys.stdout.flush()

    def run(self):

        best_rewards = []
        start_time = time()

        #     agent_list = [
        #     agents.SimpleAgent(),
        #     agents.RandomAgent(),
        #     agents.SimpleAgent(),
        #     agents.RandomAgent(),
        #     # agents.DockerAgent("pommerman/simple-agent", port=12345),
        # ]

        agent_list = [
            agents.SimpleAgent(),
            agents.SimpleAgent(),
            agents.SimpleAgent(),
            agents.SimpleAgent(),
            # agents.DockerAgent("pommerman/simple-agent", port=12345),
        ]
        env = pommerman.make('PommeFFACompetition-v0', agent_list)

        print(self.env_name)

        #run self.loops episode
        for loop in range(self.loops):
            state = env.reset()
            done = False
            root = Node(None, None)

            best_actions = []
            best_reward = float("-inf")
            #each episode, apply mcts to first simple agent
            for _ in range(self.playouts):
                env.render()
                action = env.act(state)
                state, reward, done, info = env.step(action)

                sum_reward = 0
                node = root
                terminal = False
                actions = []

                # selection
                while node.children:
                    if node.explored_children < len(node.children):
                        #expand not fully expanded node
                        child = node.children[node.explored_children]
                        node.explored_children += 1
                        node = child
                    else:
                        #best child using ucb
                        node = max(node.children, key=ucb)
                        action = env.act(state)
                        state, reward, done, info = env.step(action)
                    sum_reward += reward[0]
                    actions.append(node.action)

                # expansion
                if not done:
                    node.children = [Node(node, a) for a in combinations(env.action_space)]
                    random.shuffle(node.children)

                # playout (simulation)
                while not done:
                    action = env.act(state)
                    #act = actions.sample()
                    state, reward, done, info = env.step(action)
                   # _, reward, terminal, _ = state.step(action)
                    sum_reward += reward[0]
                    actions.append(action)

                    if len(actions) > self.max_depth:
                        sum_reward -= 100
                        break

                # remember best
                if best_reward < sum_reward:
                    best_reward = sum_reward
                    best_actions = actions

                # backpropagate
                while node:
                    node.visits += 1
                    node.value += sum_reward
                    node = node.parent

                # fix monitors not being garbage collected
                #del state._monitor

            sum_reward = 0
            for action in best_actions:
                _, reward, done, _ = env.step(action)
                sum_reward += reward[0]
                if done:
                    break

            best_rewards.append(sum_reward)
            score = max(moving_average(best_rewards, 100))
            avg_time = (time()-start_time)/(loop+1)
            self.print_stats(loop+1, score, avg_time)
        env.monitor.close()
        print()


def main():
    Runner( env_name = 'PommeFFACompetition-v0',   loops=1000, playouts=100, max_depth=50, ).run()

if __name__ == "__main__":
    main()