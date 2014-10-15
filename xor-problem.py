from numpy import *

def xor(array1, array2):
	"""Wrapper function for numpy.logical_xor"""
	return logical_xor(array1, array2);

print(xor(array([0,1,2,3]), array([4,3,1,2])))