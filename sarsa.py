from __future__ import print_function  # kompatibilität python 2 und 3

import numpy as np
import grid_world
import mlp.KTimage as KT


class SARSA_Algorithm(object):

    def __init__(self, beta, learn_rate, weightTable, map_size):
        self.weightTable = weightTable
        self.beta = beta
        self.learn_rate = learn_rate
        self.map_size = map_size

    def sarsa(self, world):
        """SARSA algorithm"""
        world.newinit()
        s = world.get_sensor()
        # hoch, runter, rechts, links
        h = np.dot(self.weightTable, s)
        aVector, a = grid_world.getAction(h, self.beta)
        val = np.dot(self.weightTable[a], s)
        r = world.get_reward()
        duration = 0
        while not r:
            if not world.act(aVector.tolist()):
                break
            s_next = world.get_sensor()
            r = world.get_reward()
            h = np.dot(self.weightTable, s_next)
            aVector_next, a_next = grid_world.getAction(h, self.beta)
            val_next = np.dot(self.weightTable[a_next], s_next)

            if r:
                target = 1.0
            else:
                target = 0.9 * val_next

            delta = target - val
            self.weightTable += 0.1 * delta * np.outer(aVector, s)
            s[0:self.map_size] = s_next[0:self.map_size]
            val = val_next
            a = a_next
            aVector = np.copy(aVector_next)
            duration += 1

        return duration

    def getWeights(self):
        return self.weightTable

    def decideAction(self, sensordata):
        h = np.dot(self.weightTable, sensordata)
        aVector, a = grid_world.getAction(h, self.beta)
        return aVector.tolist()

if __name__ == '__main__':
    training_steps = 10000
    size_a, size_b = 10, 10
    worldObj = grid_world.world(size=(size_a, size_b))
    map_size = size_a * size_b
    weightTable = np.zeros((4, map_size))
    sarsaObject = SARSA_Algorithm(5, 0.2, weightTable, map_size)

    # ------------------- train ---------------------
    d_sum = 0
    for i in range(0, training_steps):
        duration = sarsaObject.sarsa(worldObj)
        d_sum += duration
        if i % 100 == 0:
            KT.exporttiles(sarsaObject.getWeights(), size_a, size_b, "/tmp/coco/obs_W_1_0.pgm", 1, 4)

    print('Durchschnitt Trainingsdauer pro Durchlauf: {}'
          .format(d_sum / training_steps))

    # ------------------- testen -------------------
    worldObj.newinit()
    worldObj.printWorld()
    steps = 0
    while not worldObj.get_reward():
        worldObj.act(sarsaObject.decideAction(worldObj.get_sensor()))
        steps += 1
        worldObj.printWorld()

    print("steps: {}".format(steps))
