from numpy import *

def activateNeurons(W, s_in):
	'''Activates all neurons'''
	h_out = dot(W, s_in)
	# TODO: transfer function
	return h_out

def feedForwardActivation(W_hid, W_out, s_in):
	'''Feedforward activation of the MLP'''
	h_hid = dot(W_hid, s_in)
	s_hid = h_hid # TODO: transfer function
	h_out = dot(W_out, s_hid)
	return h_out # TODO: transfer function