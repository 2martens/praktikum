#!/usr/bin/python

import math
import matplotlib.pyplot as plt
import numpy as np
import random
from multi_layer import MultiLayerNetwork


if __name__ == '__main__':

    start_point = -math.pi
    end_point = math.pi*4

    network = MultiLayerNetwork(
        layout=(1, 50, 1),
        transfer_function=MultiLayerNetwork.sigmoid_function,
        last_transfer_function=MultiLayerNetwork.direct_function)

    training_data = []
    for i in np.arange(start_point, end_point, 0.01):
        expected = np.array([np.around(math.degrees(math.sin(i)), 2)])
        data_set = [[i], expected]
        training_data.append(data_set)

    for i in range(100000):
        input, expected = random.choice(training_data)
        error = network.train(input, expected, 0.001)

    results = []
    for i in np.arange(start_point, end_point, 0.1):
        res = network.calc([i])
        results.append(res)
        print("{} -> {} expected: {}".
              format([i], res, math.degrees(math.sin(i))))

    plt.plot(results)
    plt.show()
