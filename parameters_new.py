################################################ README #########################################################

# This file contains all the parameters of the network.

#################################################################################################################
import numpy as np
import random
class param:
    scale = 0.11
    T = 100
    t_back = -30
    t_fore = 30

    n_state=36
    n_cue=35
    n_prediction= 25
    n_error=25
    n_sensory=25
    
    Pref = 0.#0
    Prest = -65#0
    Pmin = -5.0
    Pth = -50.0#25
  

    epoch = 20
    
    sigma =0.05

    A_minus = 0.1 # time difference is positive i.e negative reinforcement
    A_plus = 0.5 # 0.01 # time difference is negative i.e positive reinforcement

    tau_plus = 10
    tau_minus = 10

    finaldata=[[1,2,8,9,5],
               [6,7,3,4,0],
               [10,11,21,22,23],
               [13,14,17,18,19],
               [20,15,16,12,24]]
    list=np.array(finaldata)
    # np.random.shuffle(list)
   

if __name__=="__main__":
    a=param()
    print(type(a.list))