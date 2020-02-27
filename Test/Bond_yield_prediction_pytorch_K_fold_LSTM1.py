from torch.distributions.categorical import Categorical
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import matplotlib.pyplot as plt
from utils import *
from Model import *
import pandas as pd
from sklearn.metrics import confusion_matrix

############ Hyper parameters
SEQ_SIZE = 30 #ดูย้อนหลังกี่วัน
INPUT_SIZE = 1 
HIDDEN_SIZE = 3
NUM_LAYERS = 2
OUTPUT_SIZE = 3
learning_rate = 0.0001
num_epochs = 100001
loop = 1
k_fold = 10
output_dim = 3

########### Importing the training set

dataset_train = pd.read_csv('Bond_Yield_Dataset_1_3outputs.csv')

Y = np.array(dataset_train.iloc[:, 2])#.to_numpy()
X = np.array(dataset_train.iloc[:, 1])#.to_numpy()
Y_= one_hot_encoder(Y,output_dim)
batch_size =int((len(X)-30-1)/k_fold)

Mode = "LSTM" #['LSTM','RNN']

Results = []

# Encode the outputs
Y_= one_hot_encoder(Y,output_dim)
#############################

path = "saved_model/1_LSTM_1y.pth"

############################
if Mode == "LSTM":
    Model = LSTM(INPUT_SIZE,HIDDEN_SIZE,batch_size,OUTPUT_SIZE,NUM_LAYERS)# Classifier
    optimiser = torch.optim.Adam(Model.parameters(), lr=learning_rate)
    try:
        Model.load_model(path, Model, optimiser)
    except:
        pass
elif Mode == "RNN":
    Model = RNN(INPUT_SIZE,HIDDEN_SIZE,NUM_LAYERS,OUTPUT_SIZE) # RNN
########################
optimiser = torch.optim.Adam(Model.parameters(), lr=learning_rate)

######## normalization

#X_norm = (X -X.min())/(X.max()-X.min())

#################################

X_rms = RunningMeanStd()
X_rms.update(X)
X_norm = (X - X_rms.mean)/X_rms.var

for i in range (loop):
    X_train_ = []
    Y_train_ = []
    for i in range (len(dataset_train)):
        if (i >= (SEQ_SIZE)) and i < (len(dataset_train)-1):
            X_train_.append(X_norm[i-SEQ_SIZE:i])
            Y_train_.append(Y_[i+1])
            #Y_train_.append(Y[i])
            
    X_train_ = np.array(X_train_)
    Y_train_ = np.array(Y_train_)

    train_index_list, test_index_list = split_data_index(k_fold,X_train_,Y_train_,True,None) #Shuffle only with Normal NN
    
    for epoch in range(num_epochs):
        loss = 0
        for j in range (len(train_index_list)):
            X_train = X_train_[train_index_list[j]]
            Y_train = Y_train_[train_index_list[j]]
            X_test = X_train_[test_index_list[j]]
            Y_test = Y_train_[test_index_list[j]]
            #print(len(Y_train))
            
            X_train = torch.tensor(X_train, requires_grad=True).float()
            Y_train = torch.tensor(Y_train).long()
            X_test = torch.tensor(X_test, requires_grad=True).float()
            Y_test = torch.tensor(Y_test).long().view(1,-1,1)      
            
            if len(X_test) == batch_size:
                
                #print("X_test",X_test.shape[0])
                output = Model.forward(X_train.view(1,-1,INPUT_SIZE)) #for Classifier
                output = output.view(1,-1,1)
                Y_test = Y_test.view(1,-1,1)
                #print(output.shape)
                #print(Y_test.shape)
                loss += F.binary_cross_entropy_with_logits(output.float(),Y_test.float()) #F.mse_loss(output.float(),Y_train.float())
                #print(loss)
                if epoch % 100 == 0:
                    if Mode == "Classifier":
                        output = output.view(-1)
                        output = output.data.numpy()
                        outs = []
                        for i in range (len(output)):
                            if output[i] >=0.5:
                                outs.append(1)
                            else:
                                outs.append(0)
                        outs = np.array(outs)
                        
                    Y_test = Y_test.view(-1)
                    Y_test = Y_test.data.numpy()
                    accuracy = confusion_matrix(Y_test,outs)
                    acc = (accuracy[0][0]+accuracy[1][1])/np.sum(accuracy)
                    print("epoch = ", str(epoch), " loss for the", str(j),"batch"+" = ", loss.data.numpy()," accuray = ", acc)
                    Results.append(acc)
                    
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()      
            #print('epoch {}, loss {}'.format(epoch,loss.item()))

    path = "saved_model/1_LSTM_1y.pth"
    Model.save_model(path, epoch, Model, optimiser)
