#from config import *
import numpy as np

import torch
from torch._six import inf
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import OneHotEncoder

# if default_config['TrainMethod'] in ['PPO', 'ICM', 'RND']:
#     num_step = int(ppo_config['NumStep'])
# else:
#     num_step = int(default_config['NumStep'])
"""
use_gae = default_config.getboolean('UseGAE')
lam = float(default_config['Lambda'])
train_method = default_config['TrainMethod']
"""

def one_hot_encoder(string, n_catergory):
    string_temp = []
    temp = []
    for j in range(len(string)):
        temp = np.zeros([1,n_catergory])
        for i in range (n_catergory):
        #for i in range (-1, 1, 1):
            if i == string[j]:
                temp[0][i] = 1
        string_temp.append(temp[0])
    return np.array(string_temp)

def binary_to_others(string):
    temp = []
    #string_temp = []
    temp = np.zeros([len(string)])
    for j in range(len(string)):
        #for i in range (len(string[j])):
        k = np.argmax(string[j])
        temp[j] = k
    return np.array(temp)
            
            
            
def split_data_index(n_splits, X, Y, shuffle = True ,random_state = None):
    kf = KFold(n_splits,shuffle,random_state)
    train_index_list = []
    test_index_list = []
    for train_index, test_index in kf.split(X):
        train_index_list.append(train_index)
        test_index_list.append(test_index)
    return train_index_list, test_index_list

def make_train_data(reward, done, value, gamma, num_step, num_worker):
    discounted_return = np.empty([num_worker, num_step])

    running_add = value[:, -1]
    for t in range(num_step - 1, -1, -1):
        running_add = reward[:, t] + gamma * running_add * (1 - done[:, t])
        discounted_return[:, t] = running_add

    # For Actor
    adv = discounted_return - value[:, :-1]

    return discounted_return.reshape([-1]), adv.reshape([-1])

def make_train_data_me(reward, done, value, gamma, num_step):
    discounted_return = np.empty([1, num_step])

    #print("running_add=",value[:, -1])
    running_add = value[:, -1]
    for t in range(num_step - 1, -1, -1):
        #print(t)
        #print(reward)
        #print(reward[:, t])
        #print(done)
        #print(done[:, t])
        running_add = reward[:, t] + gamma * running_add * (1 - done[:, t])
        #print("discounted_return=",discounted_return)
        discounted_return[:, t] = running_add

    # For Actor
    adv = discounted_return - value[:, :-1]

    return discounted_return.reshape([-1]), adv.reshape([-1])


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = 0

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]#len(x) #x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        #print(self.count)
        #print(batch_count)
        tot_count = self.count + int(batch_count)

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class RewardForwardFilter(object):
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems


def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]  # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]  # dito
    return e_x / div


def global_grad_norm_(parameters, norm_type=2):
    r"""Clips gradient norm of an iterable of parameters.
    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.
    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)

    return total_norm