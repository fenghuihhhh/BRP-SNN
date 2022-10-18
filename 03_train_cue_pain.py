from wave import Wave_write
import numpy as np
from neuron import neuron
import random
from matplotlib import pyplot as plt
import imageio
from my_encode import GRF,GRF2,GRF1
from STDP import STDP,update_STDP
from parameters_new import param as new_par
import os
import time as timing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import os

# UTF-8,csv
arm = pd.read_csv('Robot_RArm.csv', header=None) 
arm.columns = ["Col1", "Col2", "Col3", "Col4", "Col5", "Col6"]
X = arm[["Col1", "Col2", "Col5"]]  
X = np.array(X)
arm_test = pd.read_csv('Robot_RArm_test.csv', header=None)  
arm_test.columns = ["Col1", "Col2", "Col3", "Col4", "Col5", "Col6"]
X_test = arm_test[["Col1", "Col2", "Col5"]]  
X_test = np.array(X_test)


vision=pd.read_csv('Robot_Vision.csv',header=None)
vision1 = np.array(vision)
vision.columns = ["Col1", "Col2", "Col3"]
Y = vision[["Col3"]] 
Y = np.array(Y)
vision_test=pd.read_csv('Robot_Vision_test.csv',header=None)
vision_test.columns = ["Col1", "Col2", "Col3"]
Y_test = vision_test[["Col3"]]  
Y_test = np.array(Y_test)


start = timing.time()
print ("start:", start)
time  = np.arange(0, new_par.T, 1)# time series


gauss_neuron = 12  
center = np.ones((gauss_neuron, 1))
width = 1 / 15

for i in range(len(center)):  
    center[i] = (2 * i - 3) / 20  
x = np.arange(0, 1, 0.0001)  

num_features = 3
num_features2 = 2

gauss_recpt_field = np.zeros((gauss_neuron, len(x)))  
for i in range(gauss_neuron):
    gauss_recpt_field[i, :] = np.exp(-(x - center[i]) ** 2 / (2 * width * width))  

def gauss_response(inputs,num_features):
    spike_time = np.zeros((gauss_neuron, num_features))
    # input: shape [1, features]
    # output: shape [gaussian neurons*features] spiking time
    for i in range(num_features):
        for j in range(gauss_neuron):
            spike_time[j, i] = gauss_recpt_field[j, inputs[i]]  #entry gauss function
    spikes = []
    for i in range(spike_time.shape[1]):
        spikes.extend(spike_time[:, i])
    return np.array(spikes)


gauss_neurons = gauss_neuron * num_features
gauss_neurons2 = gauss_neuron * num_features2

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X = (X * 10000).astype(int)  #10000
X[X == 10000] = 9999 
input_spike = np.zeros((X.shape[0], gauss_neurons))  
for i in range(X.shape[0]):
    input_spike[i, :] = gauss_response(X[i, :],num_features)
input_spike[input_spike < 0.1] = 0  
input_spike = np.around(100 * (1 - input_spike))  
input_spike[input_spike == 0] = 1
input_spike[input_spike == 100] = 0
state=[]
for i in range(90):
    aa=[]
    for j in range(36):
        if input_spike[i][j] != 0:
            number=input_spike[i][j]
            aa.append((int(number),j))
    state.append(aa)

X_test = scaler.fit_transform(X_test)
X_test = (X_test * 10000).astype(int) 
X_test[X_test == 10000] = 9999 
input_spike = np.zeros((X_test.shape[0], gauss_neurons))  
for i in range(X_test.shape[0]):
    input_spike[i, :] = gauss_response(X_test[i, :],num_features)
input_spike[input_spike < 0.1] = 0   
input_spike = np.around(100 * (1 - input_spike))  
# Adjust t = 0 firing to t = 1
input_spike[input_spike == 0] = 1
input_spike[input_spike == 100] = 0
state_test=[]
for i in range(30):
    aa=[]
    for j in range(36):
        if input_spike[i][j] != 0:
            number=input_spike[i][j]
            aa.append((int(number),j))
    state_test.append(aa)

prediction=Y
prediction_test=Y_test



layer1 = []
for i in range(new_par.n_state):
    a = neuron()
    layer1.append(a)

layer2 = []
for i in range(new_par.n_prediction):
    a = neuron()
    layer2.append(a)

layer3 = []
for i in range(new_par.n_error):
    a = neuron()
    layer3.append(a)

layer4 = []
for i in range(new_par.n_prediction):
    a = neuron()
    layer4.append(a)

layer5 = []
for i in range(new_par.n_cue):
    a = neuron()
    layer5.append(a)

# random synapse matrix
s1 = pd.read_csv('synapse_state_prediction.csv', header=None) 
s1 = np.array(s1)
synapse_state_prediction = s1
synapse_prediction_error = np.diag(np.full(25,-30))
synapse_sensory_error = np.diag(np.full(25,30))
synapse_cue_error = np.zeros((new_par.n_cue,new_par.n_error))





cue=1  # represent dangerour object close to robot
if cue==1:
    train_cue = np.array(GRF1(1, new_par.n_cue))


potential_x = []
for ii in range(new_par.n_prediction):
    potential_x.append([])
spike_x = []
for ii in range(new_par.n_prediction):
    spike_x.append([])
spike_time_x = []
for ii in range(new_par.n_prediction):
    spike_time_x.append([])

potential_y = []
for ii in range(new_par.n_error):
    potential_y.append([])
spike_y = []
for ii in range(new_par.n_error):
    spike_y.append([])
spike_time_y= []
for ii in range(new_par.n_error):
    spike_time_y.append([])

# train cue module and pain module
for m in range(100):
    print('m=',m)
    train_state = np.array(GRF2(state[4], new_par.n_state))   #should be input from robot
    train_sensory = np.array(GRF(2, new_par.n_prediction))    #should be input from robot
    for x in layer2:
        x.initial(new_par.Pth)
    for y in layer3:
        y.initial(new_par.Pth)
    for t in time:
        for i, x in enumerate(layer2):
            if (x.t_rest < t):
                fh=synapse_state_prediction[i]
                fhh=train_state[:, t]
                input1 = np.dot(synapse_state_prediction[i], np.transpose(train_state[:, t]))
                x.P = x.P + input1
            potential_x[i].append(x.P)
        for i, x in enumerate(layer2):
            s = x.check()
            spike_x[i].append(s)
            if (s == 1):
                spike_time_x[i].append((m*100)+t)
                x.t_rest = t + x.t_ref
                x.P = new_par.Prest
        spike=np.array(spike_x)
        for j, y in enumerate(layer3):
            input11 = np.dot(synapse_prediction_error[j], np.transpose(spike[:,-1]))
            input22 = np.dot(synapse_sensory_error[j], np.transpose(train_sensory[:, t]))
            y.P = y.P + input11 + input22
            potential_y[j].append(y.P)
    for j, y in enumerate(layer3):
        ss = y.check()
        spike_y[j].append(ss)
        if ss==1:                
            for h in range(new_par.n_cue):
                for t1 in range(new_par.t_back, new_par.t_fore + 1, 1):
                    if 0 <= t + t1 < new_par.T + 1 and train_cue[h][t + t1] == 1:
                            synapse_cue_error[h][j] = update_STDP(synapse_cue_error[h][j], STDP(t1))


np.savetxt('synapse_cue_error.csv', synapse_cue_error, delimiter=',')

plt.figure(1)
plt.gca().grid(False)
weights = np.array(synapse_cue_error)
img = np.zeros((new_par.n_cue,new_par.n_error))
for i in range(new_par.n_cue):
    for j in range(new_par.n_error):
        img[i][j] = np.interp(weights[i][j], [0,50], [0,50])
plt.imshow(img)
plt.colorbar()
plt.title('Weight(cue_error)')
plt.show()
plt.savefig('Weight(cue_error).png')



