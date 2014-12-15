""" Multi Layer Perceptron with SGD """

import numpy as np
import matplotlib.pyplot as plt
import random
import KTimage


class MultiLayerNetwork(object):

    """ A Multi Layer Perceptron """

    def __init__(self, layout):
        """
        Creates a new MultiLayerNetwork
        @param layout A list describing the number of neurons
                      per layer [inputsize, hid1, hid2,..,out]
        """

        super(MultiLayerNetwork, self).__init__()

        self.layout = layout
        self.num_layers = len(layout)
        # init weights
        self.biases = [np.random.randn(y, 1) for y in self.layout[1:]]
        self.weights = [np.random.randn(y, x) 
                        for x, y in zip(self.layout[:-1], self.layout[1:])]

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights

    def get_biases(self):
        return self.biases

    def set_biases(self, biases):
        self.biases = biases

    @staticmethod
    def sigmoid_function(value):
        return (1.0 / (1.0 + np.exp(-value)))

    @staticmethod
    def sigmoid_prime_function(value):
        return (MultiLayerNetwork.sigmoid_function(value) * (1.0 - MultiLayerNetwork.sigmoid_function(value)))

    @staticmethod
    def step_function(value, derivate=False):
        return 0 if value < 0 else 1

    @staticmethod
    def round2_function(value, derivate=False):
        """ rounds to 2 digits after decimal point """
        return np.around(value, 2)

    @staticmethod
    def direct_function(value, derivate=False):
        return value

    def feedforward(self, input):
        """
        Return the output of the network for the given input
        @param input The input of the network
        @return The output of the network
        """
        for bias, weight in zip(self.biases, self.weights):
            input = sigmoid_vectorize(np.dot(weight, input) + bias)
        return input

    def SGD(self, training_data, train_steps, mini_batch_size, learning_rate,
            test_data=None):
        '''Uses SGD to solve a problem'''

        if test_data: length_test = len(test_data)
        length = len(training_data)
        print(length)
        for j in xrange(train_steps):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, length, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
            if test_data:
                print("Iteration {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), length_test))
            else:
                print("Iteration {0} complete".format(j))

    def update_mini_batch(self, mini_batch, learning_rate):
        '''Updates the weights and biases for the mini batch'''
        new_biases = [np.zeros(b.shape) for b in self.biases]
        new_weights = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_biases, delta_weights = self.backprop(x, y)
            new_weights = [new_bias + delta_new_bias for new_bias, delta_new_bias in zip(new_biases, delta_biases)]
            new_biases = [new_weight + delta_new_weight for new_weight, delta_new_weight in zip(new_weights, delta_weights)]
        self.weights = [weight - (learning_rate / len(mini_batch)) * new_weight 
                        for weight, new_weight in zip(self.weights, new_weights)]
        self.biases = [bias - (learning_rate / len(mini_batch)) * new_bias 
                       for bias, new_bias in zip(self.biases, new_biases)]

    def backprop(self, x, y):
        """
        Returns a tuple of the deltas for the weights and biases.
        @param x
        @param y
        @return (delta_biases, delta_weights)
        """
        delta_biases = [np.zeros(bias.shape) for bias in self.biases]
        delta_weights = [np.zeros(weight.shape) for weight in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for bias, weight in zip(self.biases, self.weights):
            z = np.dot(weight, activation) + bias
            zs.append(z)
            activation = sigmoid_vectorize(z)
            activations.append(activation)
        # backward pass
        delta = self.vector_subtract(activations[-1], y) * sigmoid_prime_vectorize(zs[-1])
        delta_biases[-1] = delta
        delta_weights[-1] = np.dot(delta, activations[-2].transpose())
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sigmoid_prime_vector = sigmoid_prime_vectorize(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sigmoid_prime_vector
            delta_biases[-l] = delta
            delta_weights[-l] = np.dot(delta, activations[-l-1].transpose())
        return (delta_biases, delta_weights)

    def vector_subtract(self, x, y):
        '''
        For given x and y, returns (x - y)
        '''
        return (x - y)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y) 
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def saveWeights(self, filepath):
        np.save(filepath, self.weights)

    def loadWeights(self, filepath):
        try:
            self.weights = np.load(filepath)
        except IOError:
            print("no weights to load")

sigmoid_vectorize = np.vectorize(MultiLayerNetwork.sigmoid_function)
sigmoid_prime_vectorize = np.vectorize(MultiLayerNetwork.sigmoid_prime_function)