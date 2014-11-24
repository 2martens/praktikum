from multi_layer import MultiLayerNetwork
from world import world_digits

import numpy as np
import matplotlib.pyplot as plt


def print_image(img_data, row_len=8):

    print("")
    for i in range(len(img_data)):
        if (img_data[i] == 1):
            print("1", end="")
        else:
            print(" ", end="")
        if (i + 1) % row_len == 0:
            print("")
    print("")

if __name__ == '__main__':

    digits = world_digits()

    inputsize, number_of_digits = digits.dim()

    network = MultiLayerNetwork(
        layout=(inputsize, 1000, inputsize),
        transfer_function = MultiLayerNetwork.sigmoid_function,
        last_transfer_function = MultiLayerNetwork.step_function)

    filepath = "/home/dennis/test.npy"
    network.loadWeights(filepath)

    # create data and result array for training
    training_data = []
    digits.newinit()
    for i in range(number_of_digits):
        input = digits.sensor()
        if i < (number_of_digits - 1):
            digits.act()
        else:
            digits.newinit()
        expected = digits.sensor()
        data_set = [input, expected]
        training_data.append(data_set)

    errors = network.train_until_fit(
        training_data=training_data,
        train_steps=500,
        learn_rate=0.2,
        max_trains=500000)

    # save to disk
    network.saveWeights(filepath)

    # check results
    digits.newinit()
    for i in range(number_of_digits):
        inputs = digits.sensor()
        print_image(digits.sensor())
        expected = np.zeros(number_of_digits)
        expected[i] = 1

        result = network.calc(inputs)
        digits.act()

        print("next:")
        print_image(result)

    plt.plot(errors)
    plt.show()
