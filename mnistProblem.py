#!/usr/bin/python3
import mnist_loader
import numpy as np
from mlp.mlp_sgd import MultiLayerNetwork
from mnistExample.network import Network

def buildData(raw_data):
	'''Builds a network input array'''
	length = len(raw_data)
	print('Length: ' + str(length))
	finalData = []
	for i in xrange(length):
		# print('i {}'.format(i))
		x = raw_data[i][0]
		y = raw_data[i][1]
		x_final = []
		y_final = []
		x_len = len(np.atleast_1d(x))
		y_len = len(np.atleast_1d(y))
		for k in xrange(x_len):
			x_k = x[k][0]
			x_final.append(x_k)

		for j in xrange(y_len):
			y_j = y[j][0]
			y_final.append(y_j)
		finalData.append((x_final, y_final))

	print('Formatting ended')
	return finalData

if __name__ == '__main__':

	print('Loading starts')
	training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
	print('Loading ended')

	network = MultiLayerNetwork([784, 30, 10])

	net = Network([784,30,10])

	# training_data = buildData(training_data)
	# validation_data = buildData(validation_data)
	# test_data = buildData(test_data)

	# print(training_data[0])
	# print(training_data[0][0].shape)
	network.SGD(training_data, 30, 10, 3.0, test_data=test_data)
	# net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

