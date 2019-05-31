from agent.agent import Agent
from functions import *

#import sys

window_size = 10
stock_name= "^DJI"
episode_count = 5

agent = Agent(window_size*2)


Adjclose = getStockDataVec(stock_name)
Volume = getStockVolume(stock_name)
l = len(Adjclose) - 1

#aaa=Normalized(Volume.reshape(-1,1))
#getStatetest(Adjclose,Volume, 10, window_size + 1)
#Input= Adjclose+ Volume

batch_size = 32

for e in range(episode_count + 1):
	print("Episode " + str(e) + "/" + str(episode_count))
	state = getState(Adjclose,Volume, 0, window_size + 1)

	total_profit = 0
	agent.inventory = []

	for t in range(l):
		action = agent.act(state)


		if action == 1: # buy
			agent.inventory.append(Adjclose[t])
			print("Buy: " + formatPrice(Adjclose[t]))

		elif action == 2 and len(agent.inventory) > 0: # sell
			bought_price = agent.inventory.pop(0)
			reward = max(Adjclose[t] - bought_price, 0)
			total_profit += Adjclose[t] - bought_price
			print("Sell: " + formatPrice(Adjclose[t]) + " | Profit: " + formatPrice(Adjclose[t] - bought_price))

		# sit
		next_state = getState(Adjclose,Volume, t+1, window_size + 1)
		reward = 0
        
		done = True if t == l - 1 else False
		agent.memory.append((state, action, reward, next_state, done))
		state = next_state

		if done:
			print("--------------------------------")
			print("Total Profit: " + formatPrice(total_profit))
			print("--------------------------------")

		if len(agent.memory) > batch_size:
			agent.expReplay(batch_size)

	if e % 1 == 0:
		agent.model.save("models/model_ep" + str(e))
