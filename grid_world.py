#!/usr/bin/python

import numpy as np
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
        """ Expects the output from network """
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

    def get_sensor(self):
        return np.array(self.world.flat)

    def get_sensor2d(self):
        return self.world

    def get_reward(self):
        return self.agent_position == self.target


if __name__ == '__main__':
    W = world((4, 4), (2, 2))
    print(W.get_sensor2d(), "\n")
    # nach links bewegen
    W.act([0, 0, 0, 1])
    print(W.get_sensor2d())
