from presizer import PreSizer
from world_retina import world_retina
import numpy as np
import KTimage
import random
import os
import math


class EdgeDetector(object):

    """docstring for EdgeDetector"""

    FOLDER = "data"
    KTIMAGE_DATA = "/tmp/coco"

    def __init__(self, width, height):
        super(EdgeDetector, self).__init__()

        size = width * height

        self.width = width
        self.height = height

        self.outputFun = print
        self.layout = (size, size, size)

        # init weights
        self.weights = []
        # for i in range(len(self.layout) - 1):
        #     start_weights = np.random.uniform(
        #         -0.1, +0.1, (self.layout[i + 1], self.layout[i]))

        #     self.weights.append(start_weights)

        self.weights.append(np.random.uniform(-0.1, 0.1, (size, size)))
        self.weights.append(np.random.uniform(-0.1, 0.1, (size, size)))

        self.layer_transfer = EdgeDetector.transfer_function
        self.last_layer_transfer = EdgeDetector.theWinnerTakesItAll

        # sicherstellen das der Ordner für visualize vorhanden ist
        try:
            os.makedirs(EdgeDetector.KTIMAGE_DATA)
        except FileExistsError:
            pass

    def sigmoid_function(value, derivate=False):
        if not derivate:
            return (1 / (1 + np.exp(-value)))
        else:
            return (value * (1 - value))

    @staticmethod
    def transfer_function(x, derivate=False):
        a = 0.5
        b = 2.0
        if not derivate:
            #  f(x) = b (x - a x / (1 + b^2 x^2))
            return b * (x - (a * x / (1.0 + ((b ** 2) * (x ** 2)))))
        else:
            #  f'(x) = b (1 + a (b^2 x^2 - 1) / (b^2 x^2 + 1)^2)
            return b * (1.0 + (a * (((b ** 2) * (x ** 2) - 1.0) / (((b ** 2) * (x ** 2) + 1.0) ** 2))))

    @staticmethod
    def direct_function(value, derivate=False):
        return value

    def run(self, input):
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
            self.inputs.append(lastNetResult)

            # calc result
            lastNetResult = np.dot(self.weights[i], lastNetResult)
            if i == len(self.layout) - 2:
                # different activation function for last layer
                lastNetResult = np.array(list(map(
                    self.last_layer_transfer, np.nditer(lastNetResult))))
            else:
                # lastNetResult = self.layer_transfer(lastNetResult)
                lastNetResult = np.array(list(map(
                    self.layer_transfer, np.nditer(lastNetResult))))

            self.outputs.append(lastNetResult)

        return lastNetResult

    def train(self, training_data, expected, learn_rate=0.2):
        """
        Trains the network using the backpropagation algorithm
        @return error
        """

        # run the network
        self.run(training_data)

        # calc error
        # layer_errors = []
        # weigth_change = []
        error = 0.5 * sum((np.add(expected, -self.outputs[-1]) ** 2))

        # for i in reversed(range(len(self.layout) - 1)):
        # der letzte layer
        #     if i == (len(self.layout) - 2):
        #         layer_error = np.add(expected, -self.outputs[-1])
        #         layer_errors.append(layer_error)

        #         delta_weight = np.outer(
        #             layer_error, self.inputs[-1]) * learn_rate

        #         weigth_change.append(delta_weight)
        # self.weights[-1] += delta_weight
        #     else:
        #         layer_error = np.dot(layer_errors[-1],
        #                              self.weights[i + 1])
        #         layer_error *= self.layer_transfer(self.outputs[i], True)
        #         layer_errors.append(layer_error)

        #         delta_weight = np.outer(
        #             layer_error, self.inputs[i]) * learn_rate * 10

        #         weigth_change.append(delta_weight)
        # self.weights[i] += delta_weight

        # Der letzte Layer
        layer_error1 = np.add(expected, -self.outputs[1])
        delta_weight1 = np.outer(layer_error1, self.inputs[1]) * learn_rate

        # Hidden layer
        back = np.array(
            list(map(lambda x: self.layer_transfer(x, True), self.outputs[0])))
        layer_error0 = back
        layer_error0 *= np.dot(layer_error1, self.weights[1])
        delta_weight0 = np.outer(layer_error0, self.inputs[0]) * learn_rate

        # Update weights
        self.weights[0] += delta_weight0
        self.weights[1] += delta_weight1

        self.weights[0] -= self.weights[0] * learn_rate * 0.001
        self.weights[1] -= self.weights[1] * learn_rate * 0.001

        if math.isnan(self.weights[0][0][0]):
            print("---------------------------------------------")
            print(layer_error1, layer_error0, back)
            print("---------------------------------------------")

        # for i in range(len(self.layout) - 1):
        #     self.weights[i] += weigth_change[(len(self.layout) - 2) - i]
        #     self.weights[i] -= self.weights[i] * learn_rate * 0.0001

        return error

    def all_pass(self, training_data):
        """
        Returns true if the network calculates for all data the expected values
        @param training_data ([input1, input2,..], [expected1, expected...])
        """
        for data, expected in training_data:
            result = self.run(data)
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
                if trains % train_steps == 0:
                    self.visualize()

        self.outputFun("succeeded after {} trains".format(trains))
        return errors

    def visualize(self):
        # print(self.weights)
        KTimage.exporttiles(
            self.weights[0],
            self.width, self.height,
            "/tmp/coco/obs_W_{}_{}.pgm".format(0 + 1, 0),
            self.width, self.height)

        KTimage.exporttiles(
            self.weights[1].T,
            self.width, self.height,
            "/tmp/coco/obs_W_{}_{}.pgm".format(1 + 1, 1),
            self.width, self.height)

        KTimage.exporttiles(self.inputs[0], self.width, self.height,
                            "/tmp/coco/obs_S_0.pgm")
        KTimage.exporttiles(self.inputs[1], self.width, self.height,
                            "/tmp/coco/obs_S_1.pgm")
        KTimage.exporttiles(self.outputs[-1], self.width, self.height,
                            "/tmp/coco/obs_S_2.pgm")


def main():
    edge = EdgeDetector(10, 10)
    retina = world_retina('BSDS30_filt', 32, 481, 481, 10)

    datenSatz = []

    for i in range(200000):
        data = retina.sensor()
        retina.act()

        datenSatz.append([data, data])

    edge.train_until_fit(datenSatz, 500, 0.1, 50000000)

    # size = PreSizer.IMAGE_SIZE
    # edge = EdgeDetector((size, size, size))
    # files = [x for x in os.listdir(EdgeDetector.FOLDER)
    #          if x.endswith(".jpg")]
    # dataSet = []

    # for image in files:
    #     img = PreSizer.getOptimizedImage(EdgeDetector.FOLDER + "/" + image)
    #     data = PreSizer.getDataFromImage(img)

    #     expected = data

    #     dataSet.append([data, expected])

    # print("starting training")

    # print(files[0])
    # print(len(dataSet[0][0]), PreSizer.IMAGE_SIZE)
    # print(dataSet[0][0])

    # edge.train_until_fit(
    #     dataSet, 500, 0.1, 800000)

if __name__ == '__main__':
    main()


# def train(self):
#     files = [x for x in os.listdir(EdgeDetector.FOLDER)
#              if x.endswith(".jpg")]
#     dataSet = []

#     for image in files:
#         img = PreSizer.getOptimizedImage(EdgeDetector.FOLDER + "/" + image)
#         data = PreSizer.getDataFromImage(img)

#         expected = data

#         dataSet.append([data, expected])

#     self.outputFun("starting training")

#     print(files[0])
#     print(len(dataSet[0][0]), PreSizer.IMAGE_SIZE)
#     print(dataSet[0][0])

#     self.edgeNetwork.train_until_fit(
#         dataSet, 1000, 0.001, 800000)

#     self.isTrained = True
