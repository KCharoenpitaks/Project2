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
from sklearn.neural_network import MLPClassifier

import math

# number of previous lookup as a input
INPUT_SIZE = ['1','2','3','4','5','6','7','8','9','10',
              '11','12','13','14','15','16','17','18','19','20',
              '21','22','23','24','25','26','27','28','29','30',
              '31','32','33','34','35','36','37','38','39','40',
              '41','42','43','44','45','46','47','48','49','50']

#############
"""
def sigmoid(x):
  return 1 / (1 + math.exp(-x))
"""

dataset_train = pd.read_csv('Bond_Yield_Dataset.csv')

Y = np.array(dataset_train.iloc[:, 2])#.reshape(-1,1)
X = np.array(dataset_train.iloc[:, 1])#.reshape(-1,1)

#################  Normalization ##############################

X_rms = RunningMeanStd()
X_rms.update(X)
X_norm = (X - X_rms.mean)/X_rms.var
results = []


for input_sizes in INPUT_SIZE:
    X_train = []
    Y_train = []
    ################# Prepare Data
    for i in range (len(dataset_train)):
        if i- int(input_sizes)>= int(input_sizes):
            X_train.append(X_norm[i-int(input_sizes):i])
            Y_train.append(Y[i])
    
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    
    classifier = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=1)
    
    classifier.fit(X_train, Y_train)
    
    classifier.predict(X_train)
    classifier.predict_proba(X_train)
    classifier.score(X_train, Y_train)
    print("The Score for "+input_sizes  +" previous lookup is = ", classifier.score(X_train, Y_train))
    results.append(classifier.score(X_train, Y_train))



