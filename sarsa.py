import random

class SARSA_Algorithm(object):

	def __init__(self):
		self.qTable = [] # weight matrix
		self.weightTable = []
	def sarsa(self, world):
		'''SARSA algorithm'''
		world.new_init()
		s = world.get_sensor()
		# hoch, runter, rechts, links
		h = [0,0,0,0]
		for i in range(0, 4):
			h[i] = 1 # sum of all j in s over w_ij * s_j
		a = self.get_action(h)
		self.qTable[s, a] = h[a]
		r = world.get_reward()
		while r == 0:
			world.act(a)
			s_next = world.get_sensor()
			r = world.get_reward()
			h = [0,0,0,0]
			for i in range(0, 4):
				h[i] = 1 # sum of all j in s over w_ij * s_j -> dot product
			a_next = self.get_action(h)
			qTable_next = qTable
			qTable_next[s, a] = h[a]
			prediction = r # + gamma Q' - Q
			# for all i and j, do weight update
			s = s_next
			a = a_next
			self.qTable = qTable_next

if __name__ == '__main__':
	sarsaObject = SARSA_Algorithm()
	sarsaObject.sarsa()