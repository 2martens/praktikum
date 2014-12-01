import numpy as np
import grid_world


class SARSA_Algorithm(object):

    def __init__(self, beta, learn_rate, weightTable, map_size):
        self.weightTable = weightTable
        self.beta = beta
        self.learn_rate = learn_rate
        self.map_size = map_size

    def sarsa(self, world):
        '''SARSA algorithm'''
        world.newinit()
        s = world.get_sensor()
        successfull = True
        # hoch, runter, rechts, links
        h = np.dot(self.weightTable, s)
        aVector, a = grid_world.getAction(h, self.beta)
        val = np.dot(self.weightTable[a], s)
        r = world.get_reward()
        duration = 0
        while not r and successfull:
            successfull = world.act(aVector.tolist())
            s_next = world.get_sensor()
            r = world.get_reward()
            h = np.dot(self.weightTable, s_next)
            aVector, a_next = grid_world.getAction(h, self.beta)
            val_next = np.dot(self.weightTable[a_next], s_next)

            if r:
                target = 1.0

            else:
                target = 0.9 * val_next

            delta = target - val
            self.weightTable += 0.5 * delta * np.outer(aVector, s)
            s[0:self.map_size] = s_next[0:self.map_size]
            val = val_next
            a = a_next
            duration += 1

        return duration

    def getWeights(self):
        return self.weightTable

    def decideAction(self, sensordata):
        h = np.dot(self.weightTable, sensordata)
        aVector, a = grid_world.getAction(h, self.beta)
        return aVector.tolist()

if __name__ == '__main__':
    size_a, size_b = 3, 3
    worldObj = grid_world.world(size=(size_a, size_b))
    map_size = size_a * size_b
    weightTable = np.random.uniform(0.0, 0.0, (4, map_size))
    sarsaObject = SARSA_Algorithm(5, 0.2, weightTable, map_size)

    print(worldObj.get_sensor2d())

    d_sum = 0
    for i in range(0, 100):
        duration = sarsaObject.sarsa(worldObj)
        d_sum += duration
        print('weights: ')
        print(sarsaObject.getWeights())

    print('Durchschnitt duration:')
    print(d_sum / 5)

    worldObj.newinit()
    print(worldObj.get_sensor2d())
    while not worldObj.get_reward():
        worldObj.act(sarsaObject.decideAction(worldObj.get_sensor()))
        print(worldObj.get_sensor2d())
