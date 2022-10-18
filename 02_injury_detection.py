from wave import Wave_write
import numpy as np
from neuron import neuron
import random
from matplotlib import pyplot as plt
import imageio
from my_encode import GRF,GRF2
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

# data preprocess
arm = pd.read_csv('./Dataset_Nao/RArm.csv', header=None) 
arm.columns = ["Col1", "Col2", "Col3", "Col4", "Col5", "Col6"]
X = arm[["Col1", "Col2", "Col5"]]  
X = np.array(X)
arm_test = pd.read_csv('./Dataset_Nao/RArm_test.csv', header=None)  
arm_test.columns = ["Col1", "Col2", "Col3", "Col4", "Col5", "Col6"]
X_test = arm_test[["Col1", "Col2", "Col5"]]  
X_test = np.array(X_test)


vision=pd.read_csv('./Dataset_Nao/Vision.csv',header=None)
vision1 = np.array(vision)
vision.columns = ["Col1", "Col2", "Col3"]
Y = vision[["Col3"]] 
Y = np.array(Y)
vision_test=pd.read_csv('./Dataset_Nao/Vision_test.csv',header=None)
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



s1 = pd.read_csv('synapse_state_prediction.csv', header=None) 
s1 = np.array(s1)
synapse_state_prediction = s1
synapse_prediction_error = np.diag(np.full(25,-30))
synapse_sensory_error = np.diag(np.full(25,30))




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



fire_rate_x=[]
fire_rate_y=[]
for m in range(10):
    train_state = np.array(GRF2(state[0], new_par.n_state))  #should be input from robot
    train_sensory = np.array(GRF(2, new_par.n_prediction))   #should be input from robot
    for x in layer2:
        x.initial(new_par.Pth)
    for y in layer3:
        y.initial(new_par.Pth)
    for t in time:
        for i, x in enumerate(layer2):
            if (x.t_rest < t):
                
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
    


plt.figure(1)
for i in range(new_par.n_state):
    for j in range(1000):
        if train_state[i][j%100]==1:
            plt.plot(j,i,'k.')
plt.xlim(0,1000)
plt.ylim(0,new_par.n_state)
plt.xlabel('Time(ms)')
plt.ylabel('Neuron index')
plt.title('state module')

spike_x=np.array(spike_x)
plt.figure(2)
for i in range(new_par.n_prediction):
    for j in range(1000):
        if spike_x[i][j]==1:
            plt.plot(j,i,'k.')
plt.xlim(0,1000)
plt.ylim(0,new_par.n_prediction)
plt.xlabel('Time(ms)')
plt.ylabel('Neuron index')
plt.title('prediction module')
plt.show()

spike_y=np.array(spike_y)
plt.figure(3)
for i in range(new_par.n_error):
    for j in range(1000):
        if spike_y[i][j//100]==1 and j%100==0:
            plt.plot(j,i,'k.')
plt.xlim(0,1000)
plt.ylim(0,new_par.n_error)
plt.xlabel('Time(ms)')
plt.ylabel('Neuron index')
plt.title('pain module')

plt.figure(4)
for i in range(new_par.n_sensory):
    for j in range(100):
        if train_sensory[i][j]==1:
            plt.plot(j,i,'k.')
plt.xlim(0,100)
plt.ylim(0,new_par.n_sensory)
plt.xlabel('Time(ms)')
plt.ylabel('Neuron index')
plt.title('sensory module')

print('end')


