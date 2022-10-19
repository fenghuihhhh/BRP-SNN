
import numpy as np
import random
from matplotlib import pyplot as plt
from parameters_new import param as par

class neuron:
    def __init__(self):
        self.t_ref = 0
        self.t_rest = -1
        self.P = par.Prest
        self.P_STS=0
        self.Prest = par.Prest
    def check(self):
        if self.P>= self.Pth:#50
            self.P = self.Prest#0
            return 1

        else:
            return 0
    def inhibit(self):
        self.P  = par.Pmin
    def initial(self, th):

        self.t_rest = -1
        self.P = par.Prest
        self.Pth = th
