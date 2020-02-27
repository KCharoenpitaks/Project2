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
INPUT_SIZE = 1 #ดูย้อนหลังกี่วัน
HIDDEN_SIZE = 1
NUM_LAYERS = 2
OUTPUT_SIZE = 1
learning_rate = 0.0001
num_epochs = 350001
loop = 1
k_fold = 10
SEQ_SIZE = 30
########### Importing the training set

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_train = pd.read_csv('Bond_Yield_Dataset.csv')

Y = np.array(dataset_train.iloc[:, 2])#.to_numpy()
X = np.array(dataset_train.iloc[:, 1])#.to_numpy()

batch_size =int((len(X)-1-SEQ_SIZE)*0.9)+1

Mode = "Classifier" #['Classifier','RNN']

Results = []
#############################

path = "saved_model/1_LSTM_1y.pth"

############################
if Mode == "Classifier":
    Model = LSTM(INPUT_SIZE,HIDDEN_SIZE,SEQ_SIZE)# Classifier
    
    optimiser = torch.optim.Adam(Model.parameters(), lr=learning_rate)
    try:
        Model.load_model(path, Model, optimiser)
    except:
        pass
elif Mode == "RNN":
    Model = RNN(INPUT_SIZE,HIDDEN_SIZE,NUM_LAYERS,OUTPUT_SIZE) # RNN
########################
optimiser = torch.optim.Adam(Model.parameters(), lr=learning_rate)
hidden = torch.zeros(1,HIDDEN_SIZE)


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
            Y_train_.append(Y[i+1])
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
            
            X_train = torch.tensor(X_train, requires_grad=True).float().to(device)
            Y_train = torch.tensor(Y_train).long().to(device)
            X_test = torch.tensor(X_test, requires_grad=True).float().to(device)
            Y_test = torch.tensor(Y_test).long().view(1,-1,1).to(device)
            optimize = False
            if len(X_train) == batch_size:
                optimize = True
            
                #print("X_test",X_test.shape[0])
                X_train.view(-1,SEQ_SIZE,1)
                X_train.view(-1,SEQ_SIZE,1).shape
                #X_train.view(1,-1,INPUT_SIZE).shape
                #len(X_train.view(-1,SEQ_SIZE,1))
                output = Model.forward(X_train.view(-1,SEQ_SIZE,1)) #for Classifier
                output.shape
                #output = output.view(1,-1,1)
                #Y_train = Y_train.view(1,-1,1)
                #print(output.shape)
                #print(Y_train.shape)
                
                loss += F.binary_cross_entropy_with_logits(output.float(),Y_train.float()) #F.mse_loss(output.float(),Y_train.float())
                #print(loss)
                if epoch % 100 == 0:
                    if Mode == "Classifier":
                        X_test.shape
                        X_test.view(-1,INPUT_SIZE,1).shape
                        #len(X_test.view(-1,INPUT_SIZE,1))
                        output = Model.forward(X_test.view(-1,SEQ_SIZE,1))
                        #output = Model.forward(X_test.view(1,-1,SEQ_SIZE))
                        output.shape
                        output = output.view(-1)
                        output = output#.data.numpy()
                        
                        outs = []
                        for i in range (len(output)):
                            if output[i] >=0.5:
                                outs.append(1)
                            else:
                                outs.append(0)
                        outs = torch.tensor(np.array(outs))
                    
                    Y_test = Y_test.view(-1)
                    Y_test = Y_test#.data.numpy()
                    
                    accuracy = confusion_matrix(Y_test.cpu(),outs.cpu())
                    acc = (accuracy[0][0]+accuracy[1][1])/np.sum(accuracy)
                    print("epoch = ", str(epoch), " loss for the", str(j),"batch"+" = ", loss.data.cpu().numpy()," accuray = ", acc)
                    Results.append(acc)
        if optimize == True:     
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()      
            optimize = False
            #print('epoch {}, loss {}'.format(epoch,loss.item()))

    path = "saved_model/1_LSTM_1y.pth"
    Model.save_model(path, epoch, Model, optimiser)