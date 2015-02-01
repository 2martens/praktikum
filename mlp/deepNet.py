#!/usr/bin/python

from multi_layer import MultiLayerNetwork
import numpy as np
import random


class DeepNet(object):

    """
    Eine Klasse um mehrere (Funktionen) Netwerke hintereinander auszuführen.
    """

    def __init__(self, networkList=list()):
        super(DeepNet, self).__init__()

        self.networks = networkList

    # def appendFunction(self, calcFunction, trainFunction):
    #     self.networks.append({'calc': calcFunction, 'train': trainFunction})

    def appendNetwork(self, network):
        """
        Fügt ein neues netz an. Die Methoden calc(input) und
        backpropagate(error, last_layer, learn_rate) müssen vorhanden sein
        """
        self.networks.append(network)

    def calc(self, input):
        assert len(self.networks) > 0

        nextLayerInput = input
        for net in self.networks:
            nextLayerInput = net.calc(nextLayerInput)

        return nextLayerInput

    def train(self, training_data, expected, learn_rate=0.2):
        output = self.calc(training_data)
        error = np.add(expected, -output)

        for i, net in enumerate(reversed(self.networks)):
            error = net.backpropagate(error, i == 0, learn_rate)

        return error


def main():
    training_data = [
        (np.array([0, 0]), [0]),
        (np.array([0, 1]), [1]),
        (np.array([1, 0]), [1]),
        (np.array([1, 1]), [0]),
    ]

    first = MultiLayerNetwork(
        layout=(2, 2),
        transfer_function=MultiLayerNetwork.sigmoid_function,
        last_transfer_function=MultiLayerNetwork.sigmoid_function)

    second = MultiLayerNetwork(
        layout=(2, 1),
        transfer_function=MultiLayerNetwork.sigmoid_function,
        last_transfer_function=MultiLayerNetwork.step_function)

    net = DeepNet([first, second])

    errors = []
    for i in range(10000):
        data, result = random.choice(training_data)
        # print(data, result)
        errors.append(net.train(data, result, 0.2))

    for data, __ in training_data:
        result = net.calc(data)
        print("{} -> {}".format(data, result))


if __name__ == '__main__':
    main()
