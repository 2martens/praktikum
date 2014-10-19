""" Multi Layer Perceptron """

import numpy as np


class MultiLayerNetwork(object):

    """ A Multi Layer Perceptron """

    def __init__(self, layout):
        super(MultiLayerNetwork, self).__init__()
        self.layout = layout

        # init weigths
        self.weigths = []
        for i in range(len(layout) - 1):
            # plus 1 for bias
            start_weigths = np.zeros([layout[i + 1], layout[i] + 1])

            self.weigths.append(start_weigths)
        print(self.weigths)

    def _sigmoid_function(self, value):
        return (1 / (1 + np.exp(-value)))

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

    def train(self):
        pass

if __name__ == '__main__':
    network = MultiLayerNetwork((2, 7, 1))
    print(network.calc([1, 3]))
