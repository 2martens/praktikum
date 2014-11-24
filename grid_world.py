#!/usr/bin/python

import numpy as np
import math
from random import randint


class world(object):

    """docstring for world"""

    def __init__(self, size=(16, 16), target=(0, 0), startpos=None):
        super(world, self).__init__()
        self.size = np.array(size)
        self.target = target
        self.newinit(startpos)

    def newinit(self, startpos=None):
        if startpos is None:
            startpos = (
                randint(0, self.size[0] - 1), randint(0, self.size[1] - 1))
        self.world = np.zeros(self.size)
        self.world[startpos] = 1
        self.agent_position = startpos

    def position_in_world(self, position):
        return (position[0] >= 0 and position[1] >= 0 and
                position[0] < self.size[0] and
                position[1] < self.size[1])

    def act(self, action):
        """
        Expects the output from network as array with one 1 at position for
        action. Example [0, 0, 1, 0]
        """
        # hoch, runter, rechts, links
        actions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        new_pos = self.agent_position
        if 1 in action:
            new_pos = tuple(
                np.add(self.agent_position, actions[action.index(1)]))

            if self.position_in_world(new_pos):
                self.world[self.agent_position] = 0
                self.world[new_pos] = 1
                self.agent_position = new_pos
                return True

        return False


    def get_sensor(self):
        return np.array(self.world.flat)

    def get_sensor2d(self):
        return self.world

    def get_reward(self):
        return self.agent_position == self.target


def getAction(weights, beta, action_len=4):
    """
    Returns an action vector based on weights for actions
    """
    betas = [math.exp(x * beta) for x in weights]
    total = sum(betas)
    intervalle = []
    intervallstart = 0

    for i in betas:
        intervalle.append(intervallstart + (i / total))
        intervallstart = intervalle[-1]

    intervalle[-1] = 1  # sometimes 0.99999

    random_number = np.random.uniform()
    result = np.zeros(action_len)

    for idx, val in enumerate(intervalle):
        if random_number < val:
            result[idx] = 1
            break

    return result, idx

if __name__ == '__main__':
    # W = world((4, 4), (2, 2))
    # print(W.get_sensor2d(), "\n")
    # nach links bewegen
    # W.act([0, 0, 0, 1])
    # print(W.get_sensor2d())

    print(getAction([0.1, 0.2, 0.3, 0.2], 1))
