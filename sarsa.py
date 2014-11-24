import random
import numpy as np
import grid_world

class SARSA_Algorithm(object):

	def __init__(self, beta, learn_rate, weightTable, map_size):
		self.weightTable = weightTable
		self.beta = beta
		self.learn_rate = learn_rate
		self.map_size = map_size
	def sarsa(self, world):
		'''SARSA algorithm'''
		world.newinit()
		s = world.get_sensor()
		print(s)
		print(self.weightTable)
		# hoch, runter, rechts, links
		h = np.dot(self.weightTable, s)
		a = self.get_action(h, self.beta)
		aVector = np.zeros(4)
		aVector[a] = 1.0
		val = np.dot(self.weightTable[a], s)
		self.qTable[s, a] = h[a]
		r = world.get_reward()
		while r == 0:
			world.act(a)
			s_next = world.get_sensor()
			r = world.get_reward()
			h = np.dot(self.weightTable, s_next)
			a_next = self.get_action(h, self.beta)
			aVector = np.zeros(4)
			aVector[a_next] = 1.0
			val_next = np.dot(self.weightTable[a_next], s_next)

			if  r == 1.0:                                   
				target = r                                  
			else:
				target = 0.9 * val_next

			delta = target - val
			self.weightTable += 0.5 * delta * np.outer(aVector, s)
			s[0:self.map_size] = s_next[0:self.map_size]
			val = val_next
			a = a_next

if __name__ == '__main__':
	size_a, size_b = 16, 16
	worldObj = grid_world.world(size=(size_a, size_b))
	map_size = size_a * size_b
	weightTable = np.random.uniform (0.0, 0.0, (4, map_size))
	sarsaObject = SARSA_Algorithm(50, 0.2, weightTable, map_size)
	sarsaObject.sarsa(worldObj)