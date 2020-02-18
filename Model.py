from torch.distributions.categorical import Categorical
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import matplotlib.pyplot as plt
from utils import *
import pandas as pd

class Classifier(nn.Module):
    def __init__(self,inputs, hidden_size, n_layers, o_size):
        super(Classifier, self).__init__()
        self.inputs = inputs
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x_1 = F.relu(self.fc1(x))
        x_2 = F.relu(self.fc2(x_1))
        x = F.sigmoid(self.fc3(x_2))
        return x

    
 
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers,output_size):
        super(RNN, self).__init__()

        self.hidden_size = torch.zeros(1,hidden_size)

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, X):
        #print(X.shape)
        #print(hidden.shape)
        combined = torch.cat((X, self.hidden_size), 1)
        self.hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = F.sigmoid(output)#self.softmax(output)
        return output, self.hidden

    def initHidden(self):
        return torch.zeros(len(X_train),HIDDEN_SIZE)
