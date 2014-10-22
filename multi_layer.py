""" Multi Layer Perceptron """

import numpy as np
import random


class MultiLayerNetwork(object):

    """ A Multi Layer Perceptron """

    def __init__(self, layout):
        super(MultiLayerNetwork, self).__init__()
        self.layout = layout

        # init weigths
        self.weigths = []
        for i in range(len(layout) - 1):
            # plus 1 for bias
            start_weigths = np.random.uniform(
                -0.1, +0.1, (layout[i + 1], layout[i] + 1))

            self.weigths.append(start_weigths)

    def _sigmoid_function(self, value):
        return (1 / (1 + np.exp(-value)))

    def _sigmoid_diff(self, value):
        return (value * (1 - value))

    def _activation_function(self, value):
        return 0 if value < 0 else 1

    def calc(self, input):

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
                lastNetResult = np.array(list(map(
                    self._activation_function, np.nditer(lastNetResult))))
            else:
                lastNetResult = self._sigmoid_function(lastNetResult)

            self.outputs.append(lastNetResult)

        return lastNetResult

    def train(self, training_data, expected, learning_rate=0.2):

        # run the network
        self.calc(training_data)

        # calc error
        deltas = []
        weigth_change = []
        error = 0.5 * sum((np.add(expected, -self.outputs[-1]) ** 2))

        for i in reversed(range(len(self.layout) - 1)):
            if i == (len(self.layout) - 2):
                layer_error = np.add(expected, -self.outputs[-1])
                deltas.append(layer_error)

                delta_w = np.outer(
                    layer_error, self.inputs[-1]) * learning_rate

                weigth_change.append(delta_w)
                #self.weigths[-1] += delta_w
            else:
                layer_error = (np.dot(deltas[-1], self.weigths[i + 1])[:-1])
                layer_error *= self._sigmoid_diff(self.outputs[i])

                deltas.append(layer_error)

                delta_w = np.outer(
                    layer_error, self.inputs[i]) * learning_rate

                weigth_change.append(delta_w)
                #self.weigths[i] += delta_w

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
            if result != expected:
                return False

        return True

    def train_until_fit(self, training_data, train_steps=1000, learn_rate=0.2):
        """
        trains the network untill all training_data passes
        @param training_data ([input1, input2,..], [expected1, expected...])
        @param train_steps on failled train train_steps times
        @param learn_rate rate with wich the network will leran
        """
        trains = 0
        while not self.all_pass(training_data):
            if trains > 0 and trains % 1000 == 0:
                print("failed on {} trains".format(trains))
            for i in range(train_steps):
                input, expected = random.choice(training_data)
                self.train(input, expected, learn_rate)
                trains += 1

        print("succeeded after {} trains".format(trains))
        return trains

if __name__ == '__main__':

    training_data = [
        (np.array([0, 0]), [0]),
        (np.array([0, 1]), [1]),
        (np.array([1, 0]), [1]),
        (np.array([1, 1]), [0]),
    ]

    network = MultiLayerNetwork((2, 4, 1))

    network.train_until_fit(training_data, 1000, 0.2)

    # print(network.weigths)

    for data, __ in training_data:
        result = network.calc(data)
        print("{} -> {}".format(data[:2], result))
