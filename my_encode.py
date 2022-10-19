import numpy as np
import math
from matplotlib import pyplot as plt
from parameters_new import param as new_par

T = 1
dt = 0.005
time  = np.arange(0, T+dt, dt)


def GRF(input,n_neuron):
	input_encode = []
	a=int(input)
	a=a-1
	for i in range(25):
		if i in new_par.list[a]:
			temp = np.ones([101, ])
			input_encode.append(temp)
		else:
			temp = np.zeros([101, ])
			input_encode.append(temp)

	return input_encode


def GRF1(input,n_neuron):
	input_encode = []
	a=int(input)
	if a==1:
		for i in range(35):
			if i >17:
				temp = np.ones([101, ])
				input_encode.append(temp)
			else:
				temp = np.zeros([101, ])
				input_encode.append(temp)

	return input_encode



def GRF2(input,n_neuron):
	a=len(input)
	input_encode = []
	for i in range(n_neuron):
		temp = np.zeros([101, ])
		input_encode.append(temp)
	for j in range(a):
		s=input[j][0]
		n=input[j][1]
		input_encode[n][s]=1

	return input_encode




