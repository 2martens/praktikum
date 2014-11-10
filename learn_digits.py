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


def interpret_result(out_array):

    out = list(out_array)
    if 1 in out_array:
        return world_digits.digits[out.index(1)]
    else:
        return "not classified"

if __name__ == '__main__':

    digits = world_digits()

    inputsize, number_of_digits = digits.dim()

    network = MultiLayerNetwork(
        layout=(inputsize, 1000, number_of_digits),
        transfer_function = MultiLayerNetwork.sigmoid_function,
        last_transfer_function = MultiLayerNetwork.step_function)

    # create data and result array for training
    training_data = []
    digits.newinit()
    for i in range(number_of_digits):
        expected = np.zeros(number_of_digits)
        expected[i] = 1
        data_set = [digits.sensor(), expected]
        training_data.append(data_set)
        digits.act()

    errors = network.train_until_fit(
        training_data=training_data,
        train_steps=500,
        learn_rate=0.2,
        max_trains=500000)

    # check results
    digits.newinit()
    for i in range(number_of_digits):
        inputs = digits.sensor()
        print_image(digits.sensor())
        expected = np.zeros(number_of_digits)
        expected[i] = 1

        result = network.calc(inputs)
        digits.act()

        print("result: {}".format(interpret_result(result)))

    plt.plot(errors)
    plt.show()
