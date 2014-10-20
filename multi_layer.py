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
            start_weigths = np.random.uniform(-
                                              0.1, +0.1, (layout[i + 1], layout[i] + 1))

            self.weigths.append(start_weigths)
   #     print(self.weigths)

    def _sigmoid_function(self, value):
        return (1 / (1 + np.exp(-value)))

    def _sigmoid_diff(self, value):
        #sig = self._sigmoid_function(value)
        return (value * (1 - value))

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
            lastNetResult = self._sigmoid_function(lastNetResult)

            self.outputs.append(lastNetResult)

        return lastNetResult

    def train(self, training_data, expected, learning_rate=0.2):

        # run the network
        self.calc(training_data)

        # calc error
        deltas = []
        weigth_change = []
        #error = 0.5 * sum((np.add(expected, -self.outputs[-1]) ** 2))

        for i in reversed(range(len(self.layout) - 1)):
            if i == (len(self.layout) - 2):
                schicht_fehler = np.add(expected, -self.outputs[-1])
                deltas.append(schicht_fehler)

                delta_w = np.outer(
                    schicht_fehler, self.inputs[-1]) * learning_rate

                weigth_change.append(delta_w)
                #self.weigths[-1] += delta_w
            else:
                schicht_fehler = (
                    np.dot(deltas[-1], self.weigths[i + 1])[:-1]) * self._sigmoid_diff(self.outputs[i])

                deltas.append(schicht_fehler)

                delta_w = np.outer(
                    schicht_fehler, self.inputs[i]) * learning_rate

                weigth_change.append(delta_w)
                #self.weigths[i] += delta_w

        # Update weights
        for i in range(len(self.layout) - 1):
            self.weigths[i] += weigth_change[(len(self.layout) - 2) - i]

if __name__ == '__main__':
    training_data = [
        (np.array([0, 0]), [0]),
        (np.array([0, 1]), [1]),
        (np.array([1, 0]), [1]),
        (np.array([1, 1]), [0]),
    ]

    network = MultiLayerNetwork((2, 7, 1))

    for i in range(100000):
        data, result = random.choice(training_data)
        network.train(data, result)

    print(network.weigths)

    for data, __ in training_data:
        result = network.calc(data)
        print("{} -> {}".format(data[:2], result))
