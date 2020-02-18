from torch.distributions.categorical import Categorical
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import matplotlib.pyplot as plt
from utils import *
from Model import *
import pandas as pd

INPUT_SIZE = 60
HIDDEN_SIZE = 64
NUM_LAYERS = 2
OUTPUT_SIZE = 1

# Hyper parameters

learning_rate = 0.001
num_epochs = 50

# Importing the training set
dataset_train = pd.read_csv('Bond_Yield_Dataset.csv')

Y = dataset_train.iloc[:, 2]
X = dataset_train.iloc[:, 1]

Model = RNN(INPUT_SIZE,HIDDEN_SIZE,NUM_LAYERS,OUTPUT_SIZE)
