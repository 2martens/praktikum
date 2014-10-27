""" Multi Layer Perceptron """

import numpy as np
import random
import matplotlib.pyplot as plt

class MultiLayerNetwork(object):

    """ A Multi Layer Perceptron """

    def __init__(self, layout, **options):
        """
        Creates a new MultiLayerNetwork
        @param layout A tuple describing the number of neurons
                      per layer (inputsize, hid1, hid2,..,out)
        Extra options:
        transfer_function
        last_transfer_function
        """

        super(MultiLayerNetwork, self).__init__()

        self.layout = layout
        self.layer_transfer = MultiLayerNetwork.sigmoid_function
        self.last_layer_transfer = MultiLayerNetwork.step_function

        if "transfer_function" in options:
            self.layer_transfer = options.get("transfer_function")
            self.last_layer_transfer = options.get("transfer_function")

        if "last_transfer_function" in options:
            self.last_layer_transfer = options.get("last_transfer_function")

        # init weigths
        self.weigths = []
        for i in range(len(layout) - 1):
            # plus 1 for bias
            start_weigths = np.random.uniform(
                -0.1, +0.1, (layout[i + 1], layout[i] + 1))
            self.weigths.append(start_weigths)

    @staticmethod
    def sigmoid_function(value, derivate=False):
        if not derivate:
            return (1 / (1 + np.exp(-value)))
        else:
            return (value * (1 - value))

    @staticmethod
    def step_function(value, derivate=False):
        return 0 if value < 0 else 1

    @staticmethod
    def round2_function(value, derivate=False):
        """ rounds to 2 digits after decimal point """
        return np.around(value, 2)

    @staticmethod
    def round_function(value, derivate=False):
        """ rounds to 1 digits after decimal point """
        tmpVal = np.around(value, 1)
        if (tmpVal == 0.9):
            tmpVal = 1
        elif (tmpVal == 0.1):
            tmpVal = 0

        return tmpVal

    @staticmethod
    def direct_function(value, derivate=False):
        """ returns the input """
        return value

    def calc(self, input):
        """
        Calculates the network output for the given input
        @param input A array of inputs [in1, in2,..]
        """

        lastNetResult = np.array(input)
        # save each layer in/output for training
        self.inputs = []
        self.outputs = []

        for i in range(len(self.layout) - 1):
            # append bias
            lastNetResult = np.hstack((lastNetResult, [1]))

            self.inputs.append(lastNetResult)

            # calc result
            lastNetResult = np.dot(self.weigths[i], lastNetResult)
            if i == len(self.layout) - 2:
                # different activation function for last layer
                lastNetResult = np.array(list(map(
                    self.last_layer_transfer, np.nditer(lastNetResult))))
            else:
                lastNetResult = self.layer_transfer(lastNetResult)

            self.outputs.append(lastNetResult)

        return lastNetResult

    def train(self, training_data, expected, learning_rate=0.2):
        """
        Trains the network using the backpropagation algorithm
        @return error
        """

        # run the network
        self.calc(training_data)
        # calc error
        layer_errors = []
        weigth_change = []
        error = 0.5 * sum((np.add(expected, -self.outputs[-1]) ** 2))

        for i in reversed(range(len(self.layout) - 1)):
            if i == (len(self.layout) - 2):
                layer_error = np.add(expected, -self.outputs[-1])
                layer_errors.append(layer_error)

                delta_weight = np.outer(
                    layer_error, self.inputs[-1]) * learning_rate

                weigth_change.append(delta_weight)
                #self.weigths[-1] += delta_weight
            else:
                layer_error = np.dot(layer_errors[-1],
                                     self.weigths[i + 1])[:-1]
                layer_error *= self.layer_transfer(self.outputs[i], True)

                layer_errors.append(layer_error)

                delta_weight = np.outer(
                    layer_error, self.inputs[i]) * learning_rate

                weigth_change.append(delta_weight)
                #self.weigths[i] += delta_weight

        # Update weights
        for i in range(len(self.layout) - 1):
            self.weigths[i] += weigth_change[(len(self.layout) - 2) - i]
        return error

    def all_pass(self, training_data):
        """
        Returns true if the network calculates for all data the expected values
        @param training_data ([input1, input2,..], [expected1, expected...])
        """
        for data, expected in training_data:
            result = self.calc(data)
            if not np.array_equal(result, expected):
                return False

        return True

    def train_until_fit(self, training_data, train_steps=1000,
                        learn_rate=0.2, max_trains=500000):
        """
        trains the network untill all training_data passes
        @param training_data ([input1, input2,..], [expected1, expected...])
        @param train_steps on failled train train_steps times
        @param learn_rate rate with wich the network will leran
        @param max_trains max trains of network or 0
        @return the list of errors while training
        """

        errors = []
        trains = 0
        while not self.all_pass(training_data):
            if trains > 0 and trains % train_steps == 0:
                print("failed on {} trains with error {}"
                      .format(trains, errors[-1]))
                if max_trains != 0 and trains >= max_trains:
                    return errors

            for i in range(train_steps):
                input, expected = training_data[i % len(training_data)]
                err = self.train(input, expected, learn_rate)
                errors.append(err)
                trains += 1

        print("succeeded after {} trains".format(trains))
        return errors

if __name__ == '__main__':

    training_data = [
        (np.array([0, 0]), [0]),
        (np.array([0, 1]), [1]),
        (np.array([1, 0]), [1]),
        (np.array([1, 1]), [0]),
    ]

    network = MultiLayerNetwork(
        layout=(2, 2, 2, 2, 1),
        last_transfer_function=MultiLayerNetwork.step_function)

    errors = []

    # data, result = random.choice(training_data)
    # network.train(data, result, 1)

    errors = network.train_until_fit(training_data, 1000, 0.2, 500000)

    # for i in range(10000):
    #     data, result = random.choice(training_data)
    #     errors.append(network.train(data, result, 0.2))

    # print(network.weigths)

    for data, __ in training_data:
        result = network.calc(data)
        print("{} -> {}".format(data, result))

    plt.plot(errors)
    plt.show()
