""" Multi Layer Perceptron """

import numpy as np
import matplotlib.pyplot as plt
import random
import KTimage
import os


class MultiLayerNetwork(object):

    """ A Multi Layer Perceptron """

    KTIMAGE_DATA = "/tmp/coco"

    def __init__(self, layout, **options):
        """
        Creates a new MultiLayerNetwork
        @param layout A tupel describing the number of neurons
                      per layer [inputsize, hid1, hid2,..,out]
        Extra options:
        transfer_function       - default sigmoid_function
        last_transfer_function  - default step_function
        output_function         - default print
        """

        super(MultiLayerNetwork, self).__init__()

        self.layout = layout
        self.numWeights = len(layout) - 1
        self.layer_transfer = MultiLayerNetwork.sigmoid_function
        self.last_layer_transfer = MultiLayerNetwork.step_function

        self.outputFun = print

        if "transfer_function" in options:
            self.layer_transfer = options.get("transfer_function")
            self.last_layer_transfer = options.get("transfer_function")

        if "last_transfer_function" in options:
            self.last_layer_transfer = options.get("last_transfer_function")

        if "output_function" in options:
            self.outputFun = options.get("output_function")

        # init weights
        self.weights = []
        for i in range(self.numWeights):
            # plus 1 for bias
            start_weights = np.random.uniform(
                -0.1, +0.1, (layout[i + 1], layout[i] + 1))

            self.weights.append(start_weights)

        # sicherstellen das der Ordner für visualize vorhanden ist
        try:
            os.makedirs(MultiLayerNetwork.KTIMAGE_DATA)
        except FileExistsError:
            pass

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights

    @staticmethod
    def sigmoid_function(value, derivate=False):
        if not derivate:
            def f(x): return (1 / (1 + np.exp(-x)))
        else:
            def f(x): return (x * (1 - x))

        return np.array(list(map(f, value)))

    @staticmethod
    def theWinnerTakesItAll(value, derivate=False):
        if not derivate:
            index = np.argmax(value)
            res = np.zeros(value.shape)
            res[index] = 1
            return res
        else:
            print("there is no derivate!")

    @staticmethod
    def step_function(value, derivate=False):
        if not derivate:
            def f(x): return 0 if x < 0 else 1
            return np.array(list(map(f, value)))
        else:
            print("there is no derivate!")

    @staticmethod
    def round2_function(value, derivate=False):
        """ rounds to 2 digits after decimal point """
        if not derivate:
            def f(x): return np.around(x, 2)
            return np.array(list(map(f, value)))
        else:
            print("there is no derivate!")

    @staticmethod
    def direct_function(value, derivate=False):
        if not derivate:
            return value
        else:
            return np.ones(value.shape)

    def calc(self, input):
        """
        Calculates the network output for the given input
        @param input A array of inputs [in1, in2,..]
        @return lastNetResult
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
            lastNetResult = np.dot(self.weights[i], lastNetResult)
            if i == len(self.layout) - 2:
                # different activation function for last layer
                lastNetResult = self.last_layer_transfer(lastNetResult)
            else:
                lastNetResult = self.layer_transfer(lastNetResult)

            self.outputs.append(lastNetResult)

        return lastNetResult

    def visualize(self):
        # print(self.weights)
        for i in range(self.numWeights):
            KTimage.exporttiles(
                self.weights[i],
                self.layout[i + 1], self.layout[i] + 1,
                MultiLayerNetwork.KTIMAGE_DATA + "/obs_W_{}_{}.pgm".
                format(i + 1, i))

    def train(self, training_data, expected, learn_rate=0.2):
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
            # der letzte layer
            if i == (len(self.layout) - 2):
                layer_error = np.add(expected, -self.outputs[-1])
                layer_errors.append(layer_error)

                delta_weight = np.outer(
                    layer_error, self.inputs[-1]) * learn_rate

                weigth_change.append(delta_weight)
            else:
                layer_error = np.dot(layer_errors[-1],
                                     self.weights[i + 1])[:-1]
                layer_error *= self.layer_transfer(self.outputs[i], True)
                layer_errors.append(layer_error)

                delta_weight = np.outer(
                    layer_error, self.inputs[i]) * learn_rate

                weigth_change.append(delta_weight)

        # Update weights
        for i in range(len(self.layout) - 1):
            self.weights[i] += weigth_change[(len(self.layout) - 2) - i]

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
                avg_error = sum(
                    errors[-len(training_data):]) / len(training_data)
                self.outputFun("failed on {} trains with error {}"
                               .format(trains, avg_error))
                if max_trains != 0 and trains >= max_trains:
                    return errors

            for i in range(train_steps):
                # input, expected = training_data[i % len(training_data)]
                input, expected = random.choice(training_data)
                errors.append(self.train(input, expected, learn_rate))
                trains += 1

        self.outputFun("succeeded after {} trains".format(trains))
        return errors

    def saveWeights(self, filepath):
        np.save(filepath, self.weights)
        self.outputFun("weights saved")

    def loadWeights(self, filepath):
        """
        Lädt die gespeicherten Gewichte aber nich das Layout des Netzes.
        Dieses muss vorher erstellt werden
        """
        succeeded = False
        try:
            self.weights = np.load(filepath)
            succeeded = True
            self.outputFun("weights loaded")
        except IOError:
            self.outputFun("no weights to load")

        return succeeded

if __name__ == '__main__':

    training_data = [
        (np.array([0, 0]), [0]),
        (np.array([0, 1]), [1]),
        (np.array([1, 0]), [1]),
        (np.array([1, 1]), [0]),
    ]

    network = MultiLayerNetwork(
        layout=(2, 2, 1),
        transfer_function=MultiLayerNetwork.sigmoid_function,
        last_transfer_function=MultiLayerNetwork.step_function)

    errors = network.train_until_fit(training_data, 1000, 0.2, 500000)

    for data, __ in training_data:
        result = network.calc(data)
        print("{} -> {}".format(data, result))

    plt.plot(errors)
    plt.show()
