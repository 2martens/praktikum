#!/usr/bin/python3
import mnist_loader
from mlp import multi_layer

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
network = MultiLayerNetwork(
	layout=(784, 30, 10)
)
network.SGD(training_data, 5000, 100, 0.2)
