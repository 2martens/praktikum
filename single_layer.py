""" Single Layer Perceptron """

import numpy as np
import random


class SingleLayerNetwork(object):

    """ A Single Layer Perceptron """

    learnFactor = 0.2

    def __init__(self, start_weights):
        super(SingleLayerNetwork, self).__init__()
        self._weights = start_weights

    def _transfer_function(self, value):
        return 0 if value < 0 else 1

    def calc(self, input):
        result = np.dot(self._weights, input)
        return self._transfer_function(result)

    def train(self, input, expected):
        error = expected - self.calc(input)
        self._weights += self.learnFactor * error * input

if __name__ == '__main__':

    # Logisches ODER
    # (input1, input2, dummyData), result
    training_data = [
        (np.array([0, 0, 1]), 0),
        (np.array([0, 1, 1]), 1),
        (np.array([1, 0, 1]), 1),
        (np.array([1, 1, 1]), 1),
    ]

    number_of_runs = 1000
    network = SingleLayerNetwork(np.random.rand(3))

    # train network
    for i in range(number_of_runs):
        data, result = random.choice(training_data)
        network.train(data, result)

    # print results
    for data, __ in training_data:
        result = network.calc(data)
        print("{} -> {}".format(data[:2], result))
