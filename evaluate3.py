import keras
from keras.models import load_model

from agent.agent import Agent
from functions import *
#import sys
"""
if len(sys.argv) != 3:
	print("Usage: python evaluate.py [stock] [model]")
	exit()
"""
"""
stock_name, model_name = sys.argv[1], sys.argv[2]
""" 
for n in range(5):
    stock_name = "^DJI"
    model_name = "model_ep"+str(n)
    model = load_model("models/" + model_name)
    window_size = int(20/2)#model.layers[1].input.shape.as_list()[0]
    

    agent = Agent(window_size, True, model_name)
    Adjclose = getStockDataVec(stock_name)
    Volume = getStockVolume(stock_name)
    l = len(Adjclose) - 1
    batch_size = 32
    
    state = getState(Adjclose,Volume, 0, window_size + 1)
    total_profit = 0
    agent.inventory = []
    
    for t in range(l):
    	action = agent.act(state)

    
    	# sit
    	next_state = getState(Adjclose,Volume, t+1, window_size + 1)
    	reward = 0; print(action)
    
    	if action == 1: # buy
    		agent.inventory.append(Adjclose[t])
    		print("Buy: " + formatPrice(Adjclose[t]))
    
    	elif action == 2 and len(agent.inventory) > 0: # sell
    		bought_price = agent.inventory.pop(0)
    		reward = max(Adjclose[t] - bought_price, 0)
    		total_profit += Adjclose[t] - bought_price
    		print("Sell: " + formatPrice(Adjclose[t]) + " | Profit: " + formatPrice(Adjclose[t] - bought_price))
    
    	done = True if t == l - 1 else False
    	agent.memory.append((state, action, reward, next_state, done))
    	state = next_state
    
    	if done:
    		print("--------------------------------")
    		print(stock_name + " Total for model"+str(n)+" Profit: " + formatPrice(total_profit))
    		print("--------------------------------")
