from multi_layer import MultiLayerNetwork

import numpy as np
import random
# import matplotlib.pyplot as plt

if __name__ == '__main__':
    network = MultiLayerNetwork(
        layout=(4, 100, 4),
        last_transfer_function=MultiLayerNetwork.direct_function)

    data = np.loadtxt("myYoubot/Youbot_joints-vision.dat")
    inputs = []
    expected = []

    # print(data.shape)

    for i in range(len(data)):
        inputs.append(data[i][:4])
        expected.append(data[i][4:])

    # training_data = [inputs, expected]
    # print(training_data)
    # network.train_until_fit(training_data)

    for i in range(100000):
        data_set = random.randint(0, len(data)-1)
        err = network.train(inputs[data_set], expected[data_set], 0.05)
        if i % 100 == 0:
            print(err)
