# This file implements STDP curve and weight update rule

import numpy as np
from matplotlib import pyplot as plt
from parameters_new import param as par

#STDP reinforcement learning curve
def STDP(t):

	if t>0:
		return -par.A_minus*np.exp(-float(t)/par.tau_plus)
	if t<=0:
		return par.A_plus*np.exp(float(t)/par.tau_minus)

def anti_STDP(t):

	if t>0:
		return par.A_minus*np.exp(-float(t)/par.tau_plus)
	if t<=0:
		return -par.A_plus*10*np.exp(float(t)/par.tau_minus)

def update_STDP(w, del_w):
    w = w + par.sigma * del_w*par.scale 
    if w>5:
        w=5
    if w<0:
        w=0
    return w
if __name__ == '__main__':
   
    print(update_STDP(0,STDP(-10)))
    print(update_STDP(0,STDP(-9)))
    print(update_STDP(0,STDP(-8)))
    print(update_STDP(0,STDP(-7)))
    print(update_STDP(0,STDP(-6)))
    print(update_STDP(0,STDP(-5)))
    print(update_STDP(0,STDP(-4)))
    print(update_STDP(0,STDP(-3)))
    print(update_STDP(0,STDP(-2)))
    print(update_STDP(0,STDP(-1)))
    print(update_STDP(0,STDP(-0)))
    print(update_STDP(0,STDP(1)))
    print(update_STDP(0,STDP(2)))
    print(update_STDP(0,STDP(3)))
    print(update_STDP(0,STDP(4)))
    print(update_STDP(0,STDP(5)))
    print(update_STDP(0,STDP(6)))
    print(update_STDP(0,STDP(7)))
    print(update_STDP(0,STDP(8)))
    print(update_STDP(0,STDP(9)))
    print(update_STDP(0,STDP(10)))

