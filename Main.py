from Agent import Agent, QNetwork, Agent_subgoal, ActorCritic, Q_table_agent_selection_agent
from Monitor import interact, interact1 
import gym
import numpy as np
from taxi import TaxiEnv
import tensorflow as tf
from memory import Memory
from frozen_lake import FrozenLakeEnv



env = FrozenLakeEnv()
#env = TaxiEnv()


agent1 = Agent()
agent2 = Agent()
agent3 = Agent()
agent4 = Agent()
agent_selection = Q_table_agent_selection_agent()
agent = Agent_subgoal()
#agent = QNetwork()
#agent = ActorCritic()

avg_rewards, best_avg_reward = interact(env, agent1,agent2,agent3,agent4,agent_selection, num_episodes=50000)

avg_rewards_single, best_avg_reward_single = interact1(env, agent, num_episodes=50000)

print("Multi-Agent average reward =", max(avg_rewards), "Single-Agent average reward =", max(avg_rewards_single))