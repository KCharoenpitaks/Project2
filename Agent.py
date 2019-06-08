import numpy as np
from collections import defaultdict
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Input
from keras.layers.merge import Add, Multiply
import keras.backend as K
from keras.optimizers import Adam
from taxi import TaxiEnv


import random
from frozen_lake import FrozenLakeEnv
 
env = FrozenLakeEnv()
#env = TaxiEnv()

class Agent_subgoal:

    def __init__(self, nA=env.action_space.n,subgoal=3, epsilon=0.04, alpha=0.05, gamma=1.0):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.subgoal=subgoal
        self.Q = defaultdict(lambda: np.zeros((subgoal,self.nA)))
        #print(self.Q[0][0])
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def select_action(self, state,subgoal, i_episode):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        # return np.random.choice(self.nA)
        self.epsilon = 1.0 / ((i_episode / 800) + 1)

        policy = np.ones(self.nA) * self.epsilon / self.nA
        policy[np.argmax(self.Q[state][subgoal])] = 1 - self.epsilon + self.epsilon / self.nA
        return np.random.choice(np.arange(self.nA), p=policy)

    def step(self, state, action,subgoal, reward, next_state, done, i_episode):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        # self.Q[state][action] += 1
        self.epsilon = 1.0 / ((i_episode / 800) + 1)

        next_policy = np.ones(self.nA) * self.epsilon / self.nA
        next_policy[np.argmax(self.Q[state][subgoal])] = 1 - self.epsilon + self.epsilon / self.nA

        self.Q[state][subgoal][action] = self.Q[state][subgoal][action] + self.alpha * (reward + self.gamma * np.sum(self.Q[next_state][subgoal] * next_policy) - self.Q[state][subgoal][action])
        

class Q_table_agent_selection_agent:

    def __init__(self, nA=4,subgoal=0, epsilon=0.04, alpha=0.05, gamma=1.0):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.subgoal=subgoal
        #self.Q = defaultdict(lambda: np.zeros((subgoal,self.nA)))
        self.Q = np.zeros([16,4])
        #print(self.Q)
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def select_action(self, state,subgoal, i_episode):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        # return np.random.choice(self.nA)
        self.epsilon = 1.0 / ((i_episode / 800) + 1)

        policy = np.ones(self.nA) * self.epsilon / self.nA
        #print("here2",policy)
        #print("here",self.Q)
        #print(state)
        #print(self.Q[state][0])
        policy[np.argmax(self.Q[state])] = 1 - self.epsilon + self.epsilon / self.nA
        return np.random.choice(np.arange(self.nA), p=policy)

    def step(self, state, action,reward, next_state, done, i_episode):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        # self.Q[state][action] += 1
        self.epsilon = 1.0 / ((i_episode / 800) + 1)
        #print("1",state)
        #print(next_state)
        
        #print(reward)
        next_policy = np.ones(self.nA) * self.epsilon / self.nA
        next_policy[np.argmax(self.Q[state])] = 1 - self.epsilon + self.epsilon / self.nA

        self.Q[state][action] = self.Q[state][action] + self.alpha * (reward + self.gamma * np.sum(self.Q[next_state] * next_policy) - self.Q[state][action])
        #print(self.Q)


class Agent:

    def __init__(self, nA=int(env.action_space.n/4)+1 , epsilon=0.04, alpha=0.05, gamma=1.0):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def select_action(self, state, i_episode):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        # return np.random.choice(self.nA)
        self.epsilon = 1.0 / ((i_episode / 800) + 1)

        policy = np.ones(self.nA) * self.epsilon / self.nA
        policy[np.argmax(self.Q[state])] = 1 - self.epsilon + self.epsilon / self.nA
        return np.random.choice(np.arange(self.nA), p=policy)

    def step(self, state, action, reward, next_state, done, i_episode):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        # self.Q[state][action] += 1
        self.epsilon = 1.0 / ((i_episode / 800) + 1)

        next_policy = np.ones(self.nA) * self.epsilon / self.nA
        next_policy[np.argmax(self.Q[state])] = 1 - self.epsilon + self.epsilon / self.nA

        self.Q[state][action] = self.Q[state][action] + self.alpha * (reward + self.gamma * np.sum(self.Q[next_state] * next_policy) - self.Q[state][action])
        
class QNetwork:
    def __init__(self, env =env , learning_rate=0.01, state_size=env.observation_space.n, 
                 action_size=env.action_space.n , hidden_size=50, 
                 name='QNetwork',epsilon=0.04, gamma = 0.99):
        
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon
        self.gamma = gamma
        self.q_network = self.build_network(self.state_size,self.action_size,learning_rate)
        self.targetq_network = self.build_network(self.state_size,self.action_size,learning_rate)
        
    def build_network(self,state_size,action_size,learning_rate):

        self.model = Sequential()

        self.model.add(Dense(64, activation='relu', input_dim=state_size))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(action_size, activation='softmax'))
        
        self.adam = Adam(lr=learning_rate, decay=0.01)#decay=0.01
        self.model.compile(loss='mse',
                      optimizer=self.adam)
        return self.model
        
        
    def select_action(self, state, i_episode):
        
        Qs = self.q_network.predict([np.array([state])])
        self.epsilon = 1.0 / ((i_episode / 1600) + 1)
        policy = np.ones(self.action_size) * self.epsilon / self.action_size
        policy[np.argmax(Qs)] = 1 - self.epsilon + self.epsilon / self.action_size
        action = np.random.choice(np.arange(self.action_size), p=policy)
        

        return action
        
    def step(self, state, action, reward, next_state, done, i_episode):
        
        x_batch, y_batch = [], []

        
        if (i_episode - 1) % 10 == 0:
            self.targetq_network = self.q_network
            
        
        targets_Qs = self.targetq_network.predict([np.array([state])])


        
        targets_Qs[0][action] = reward if done else reward + self.gamma*np.max(self.targetq_network.predict([np.array([next_state])]))

        
        x_batch.append(np.array(state))
        y_batch.append(targets_Qs[0])
        self.q_network.model.fit(np.array(x_batch), np.array(y_batch), verbose=0) 
        
        
        return self.q_network
        #model.fit(state, y_train, epochs=20, batch_size=128)
        

                    
class ActorCritic:
    def __init__(self, env =env , learning_rate=0.001, state_size=env.observation_space.n, 
                 action_size=env.action_space.n , hidden_size=50, 
                 name='QNetwork',epsilon=0.04, gamma = 0.99):
        self.learning_rate = learning_rate
		#self.epsilon = 1.0
		#self.epsilon_decay = .995
        self.epsilon = epsilon
        self.gamma = gamma
        self.tau   = 0.125
        self.state_size = state_size
        self.action_size = action_size
        self.actor_network = self.create_actor_model(self.state_size,self.action_size,learning_rate)
        self.critic_network = self.create_critic_model(self.state_size,self.action_size,learning_rate)
        
    def create_actor_model(self,state_size,action_size,learning_rate):

        self.model = Sequential()

        self.model.add(Dense(500, activation='relu', input_dim=state_size))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(1000, activation='relu'))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(500, activation='relu'))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(action_size, activation='softmax'))
        
        self.adam = Adam(lr=learning_rate, decay=0.01)
        self.model.compile(loss='mse', optimizer=self.adam)
        return self.model
    
    def create_critic_model(self,state_size,action_size,learning_rate):
                       
        self.model = Sequential()

        self.model.add(Dense(500, activation='relu', input_dim=state_size))
        #self.model.add(Dense(1000, activation='relu'))
        #self.model.add(Dropout(0.1))
        #self.model.add(Dense(500, activation='relu'))
        #self.model.add(Dropout(0.1))

        self.model.add(Dense(1))
        
        self.adam = Adam(lr=learning_rate)
        self.model.compile(loss='mse', optimizer=self.adam)
        return self.model
        
    def select_action(self, state, i_episode):
        
        Qs = self.actor_network.predict([np.array([state])])
        self.epsilon = 1.0 / ((i_episode / 1600) + 1)
        policy = np.ones(self.action_size) * self.epsilon / self.action_size
        policy[np.argmax(Qs)] = 1 - self.epsilon + self.epsilon / self.action_size
        action = np.random.choice(np.arange(self.action_size), p=policy)
        

        return action

    def step(self, state, action, reward, next_state, done, i_episode):
        
        x_batch, y_batch = [], []

        predict_reward = self.critic_network.predict([np.array([state])])
        predict_next_reward = self.critic_network.predict([np.array([next_state])])
        #print(predict_reward)
        #print(predict_next_reward)
        td_target = reward + self.gamma*predict_next_reward
        
        td_error = td_target - predict_reward
        #print(td_error)
        action_predict = self.actor_network.predict([np.array([state])])
        action_predict[0][action] = td_error + action_predict[0][action]
        
        x_batch.append(np.array(state))
        y_batch.append(action_predict[0])
        #y_batch.append(targets_Qs[0])
        self.critic_network.train_on_batch(np.array(x_batch),td_target)
        self.actor_network.train_on_batch(np.array(x_batch),np.array(y_batch))
        """
        x_batch.append(np.array(state))
        y_batch.append(targets_Qs[0])
        self.q_network.model.fit(np.array(x_batch), np.array(y_batch), verbose=0) 
        """
        
        return self.actor_network
        #model.fit(state, y_train, epochs=20, batch_size=128)
        