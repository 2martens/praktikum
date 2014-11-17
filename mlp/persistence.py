#!/usr/bin/python

import numpy as np
import random
from multi_layer import MultiLayerNetwork

class MLPPersistenceManager(object):
	"""Manages the persistence of MLPs"""

	def __init__(self, training_data, layout, last_transfer_function=MultiLayerNetwork.direct_function):
		super(MLPPersistenceManager, self).__init__()
		self.training_data = training_data
		self.network = MultiLayerNetwork(
	        layout,
	        last_transfer_function=last_transfer_function)

	def train_persist(self, step_size, learn_rate, limit):
		for i in range(limit):
			input, expected = random.choice(self.training_data)
			error = self.network.train(input, expected, learn_rate)
		weights = self.network.get_weights()
		# print(weights)
		np.savetxt('mlp_persistence1.dat', weights[0], delimiter=',')
		np.savetxt('mlp_persistence2.dat', weights[1], delimiter=',')

	def load(self):
		weights1 = np.loadtxt('mlp_persistence1.dat')
		weights2 = np.loadtxt('mlp_persistence2.dat')
		self.network.set_weights([weights1,weights2])

	def calc(self, data):
		result = self.network.calc(data)
		return result

if __name__ == '__main__':
	data = np.loadtxt('youbot.dat')
	input_data = data[:10].T[4:]
	output_data = data[:10].T[:4]
	input_data = input_data.T
	# print(input_data)
	output_data = output_data.T
	training = []
	for index, val in enumerate(input_data):
		training.append((input_data[index:(index+1)][0], output_data[index:(index+1)][0]))

	mlpPers = MLPPersistenceManager(training, (4,100,4))
	mlpPers.train_persist(1000, 0.05, 100000)