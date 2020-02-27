from torch.distributions.categorical import Categorical
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import matplotlib.pyplot as plt
from utils import *
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        x = torch.sigmoid(self.fc3(x_2))
        return x
    
    def save_model(self, path, epoch, model, optimizer):
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, path)
        
    def load_model(self, path, model, optimizer):
        state = torch.load(path)
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        
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
    
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1,num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True)

        # Define the output layer
        self.linear = nn.Linear(self.batch_size, output_dim)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both 
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1))
        #print("lstm_out.shape", lstm_out.shape)
        #print("lstm_out.shape[-1]", lstm_out[-1].shape)
        #lstm_out, self.hidden = self.lstm(input.view(len(input), -1, self.input_dim))
        
        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        #y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        #print("y_pred =",self.linear(lstm_out.view(-1,self.batch_size)))
        y_pred = self.linear(lstm_out.view(-1,self.batch_size))
        y_pred = torch.sigmoid(y_pred)
        
        return y_pred.view(-1)
    
    def save_model(self, path, epoch, model, optimizer):
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, path)
        
    def load_model(self, path, model, optimizer):
        state = torch.load(path)
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])

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
        x = torch.sigmoid(self.fc3(x_2))
        return x
    
    def save_model(self, path, epoch, model, optimizer):
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, path)
        
    def load_model(self, path, model, optimizer):
        state = torch.load(path)
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        
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
    
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1,num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True).to(device)

        # Define the output layer
        self.linear = nn.Linear(self.batch_size, output_dim).to(device)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input_):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both 
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out, self.hidden = self.lstm(input_.view(len(input_), self.batch_size, -1))
        #print("lstm_out.shape", lstm_out.shape)
        #print("lstm_out.shape[-1]", lstm_out[-1].shape)
        #lstm_out, self.hidden = self.lstm(input.view(len(input), -1, self.input_dim))
        
        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        #y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        #print("y_pred =",self.linear(lstm_out.view(-1,self.batch_size)))
        y_pred = self.linear(lstm_out.view(-1,self.batch_size))
        y_pred = torch.sigmoid(y_pred)
        
        return y_pred.view(-1)
    
    def save_model(self, path, epoch, model, optimizer):
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, path)
        
    def load_model(self, path, model, optimizer):
        state = torch.load(path)
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])


