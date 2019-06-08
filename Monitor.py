from collections import deque
import sys
import math
import numpy as np
from memory import Memory
from taxi import TaxiEnv
import time
from keras.utils.np_utils import to_categorical   
from frozen_lake import FrozenLakeEnv
import random

QNetwork = False


"""
def chooseaction(a1,a2,a3,a4):
    if a1 == 0 and a2 == 0 and a3 ==0 and a4 == 0:
        action = random.choices([0,1,2,3])
    elif a1 == 0 and a2 == 0 and a3 ==0 and a4 == 1:
        action = random.choices([3])
    elif a1 == 0 and a2 == 0 and a3 ==1 and a4 == 0:
        action = random.choices([2])
    elif a1 == 0 and a2 == 0 and a3 ==1 and a4 == 1:
        action = random.choices([2,3])
    elif a1 == 0 and a2 == 1 and a3 ==0 and a4 == 0:
        action = random.choices([1])
    elif a1 == 0 and a2 == 1 and a3 ==0 and a4 == 1:
        action = random.choices([1,3])
    elif a1 == 0 and a2 == 1 and a3 ==1 and a4 == 0:
        action = random.choices([1,2])
    elif a1 == 0 and a2 == 1 and a3 ==1 and a4 == 1:
        action = random.choices([1,2,3])
    elif a1 == 1 and a2 == 0 and a3 ==0 and a4 == 0:
        action = random.choices([0])
    elif a1 == 1 and a2 == 0 and a3 ==0 and a4 == 1:
        action = random.choices([0,3])
    elif a1 == 1 and a2 == 0 and a3 ==1 and a4 == 0:
        action = random.choices([0,2])
    elif a1 == 1 and a2 == 0 and a3 ==1 and a4 == 1:
        action = random.choices([0,2,3])
    elif a1 == 1 and a2 == 1 and a3 ==0 and a4 == 0:
        action = random.choices([0,1])
    elif a1 == 1 and a2 == 1 and a3 ==0 and a4 == 1:
        action = random.choices([0,1,3])
    elif a1 == 1 and a2 == 1 and a3 ==1 and a4 == 0:
        action = random.choices([0,1,2])
    elif a1 == 1 and a2 == 1 and a3 ==1 and a4 == 1:
        action = random.choices([0,1,2,3])

    return int(action[0])
"""

def interact(env, agent1, agent2, agent3, agent4,agent_selection, num_episodes=20000, window=100):
    """ Monitor agent's performance.
    
    Params
    ======
    - env: instance of OpenAI Gym's Taxi-v1 environment
    - agent: instance of class Agent (see Agent.py for details)
    - num_episodes: number of episodes of agent-environment interaction
    - window: number of episodes to consider when calculating average rewards

    Returns
    =======
    - avg_rewards: deque containing average rewards
    - best_avg_reward: largest value in the avg_rewards deque
    """
    # initialize average rewards
    avg_rewards = deque(maxlen=num_episodes)
    # initialize best average reward
    best_avg_reward = -math.inf
    avg_reward = -math.inf
    # initialize monitor for most recent rewards
    samp_rewards = deque(maxlen=window)
    memory = Memory(max_size=20)
    batch_sample = 5
    step_total = 0
    
    # for each episode
    for i_episode in range(1, num_episodes+1):
        # begin the episode
        state = env.reset()
        step = 0
        # initialize the sampled reward
        samp_reward = 0
        #while True: #step <= 100
        while step <= 1000:
            step_total += 1
            step += 1
            
            if QNetwork == True:
                state_encode = to_categorical(state, num_classes=env.observation_space.n)
            else:
                state_encode = state
            #print("state_enconde=",state_encode)
            # agent selects an action
            action1 = agent1.select_action(state_encode,i_episode)
            action2 = agent2.select_action(state_encode,i_episode)
            action3 = agent3.select_action(state_encode,i_episode)
            action4 = agent4.select_action(state_encode,i_episode)
            #print(action1)
            #print(np.array([action1,action2,action3,action4]))
            #action_combined = np.array([int(action1),int(action2),int(action3),int(action4)])
            #action_combined = np.array([0,1,1,0])
            
            #print(action_combined)
            #np.where(action_combined[0]==1)[0][0]
            action_combined = decode(action1,action2,action3,action4)
            
            """Add agent selection q-table"""
            action_agent_selection = agent_selection.select_action(action_combined,0,i_episode)
            #print(action_agent_selection)
            
            if action_agent_selection == 0:
                action = 0
            elif action_agent_selection == 1:
                action = 1
            elif action_agent_selection == 2:
                action = 2
            elif action_agent_selection == 3:
                action = 3
            #print(action)
            
            
            #action_all = chooseaction(action1,action2,action3,action4)
            #print(action_all)
            # agent performs the selected action
            next_state, reward, done, _ = env.step(action)
			# agent performs internal updates based on sampled experience
            ### Train using this data
            """
            if done:
                next_state = None
            """  
            if QNetwork == True:
                next_state_encode = to_categorical(next_state, num_classes=env.observation_space.n)
            else:
                next_state_encode = next_state  
                
            action1_1 = agent1.select_action(next_state,i_episode)
            action2_1 = agent2.select_action(next_state,i_episode)
            action3_1 = agent3.select_action(next_state,i_episode)
            action4_1 = agent4.select_action(next_state,i_episode)
            action_combined2 = decode(action1_1,action2_1,action3_1,action4_1)
            
            
            
            #memory.add((state_encode, action1, reward, next_state_encode, done))
            #print(next_state_encode)
            
           
            agent1.step(state_encode, action1, reward, next_state_encode, done, i_episode)
            agent2.step(state_encode, action2, reward, next_state_encode, done, i_episode)
            agent3.step(state_encode, action3, reward, next_state_encode, done, i_episode)
            agent4.step(state_encode, action4, reward, next_state_encode, done, i_episode)
            agent_selection.step(action_combined,action,reward,action_combined2,done, i_episode)
            
            #env.render()
            #print(action)
            #time.sleep(0.5)
            
            #print(step)
            """
            batch = memory.sample(1)
            #print(batch[0][0])
            state1 = batch[0][0]
            action1 = batch[0][1]
            reward1 = batch[0][2]
            next_state1 = batch[0][3]
            done1 = batch[0][4]
                
            agent.step(state1, action1, reward1, next_state1, done1, i_episode)
            """
            """"
            #env.render()
            batch_sample = 5
            if step % (batch_sample) == 0:
                if memory.count >= batch_sample:
                    batch = memory.sample(batch_sample)  
                    for i in range(len(batch)):
                        state1 = batch[i][0]
                        action1 = batch[i][1]
                        reward1 = batch[i][2]
                        next_state1 = batch[i][3]
                        done1 = batch[i][4]
                        agent.step(state1, action1,0, reward1, next_state1, done1, i_episode)            
                else:
                    batch = memory.sample(1)
                    state1 = batch[0][0]
                    action1 = batch[0][1]
                    reward1 = batch[0][2]
                    next_state1 = batch[0][3]
                    done1 = batch[0][4]
                    agent.step(state1, action1, reward1, next_state1, done1, i_episode)
            """
              
            """
            if memory.count >= batch_sample:
                batch = memory.sample(batch_sample)
                states = np.array([each[0] for each in batch])
                actions = np.array([each[1] for each in batch])
                rewards = np.array([each[2] for each in batch])
                next_states = np.array([each[3] for each in batch])
                agent.step(states, actions, rewards, next_states, done, i_episode)
            else:
                batch = memory.sample(1)
                agent.step(state, action, reward, next_state, done, i_episode)
              """
            # update the sampled reward
            samp_reward += reward
            # update the state (s <- s') to next time step
            state = next_state
            if done:
                #sampled reward
                
                samp_rewards.append(samp_reward)
                env.reset()
                state, reward, done, _ = env.step(env.action_space.sample())
                break
            else:
                state = next_state

                
        if (i_episode >= 100):
            # get average reward from last 100 episodes
            avg_reward = np.mean(samp_rewards)
            # append to deque
            avg_rewards.append(avg_reward)
            # update best average reward
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                """
        if (i_episode%100 == 0):
            env.render()
            """
        # monitor progress

        print("\rEpisode {}/{} || Best average reward {} || average reward {} || episode reward {}".format(i_episode, num_episodes, best_avg_reward, avg_reward, samp_reward), end="")
        sys.stdout.flush()
        # check if task is solved (according to OpenAI Gym)
        if best_avg_reward >= 9.7:
            print('\nEnvironment solved in {} episodes.'.format(i_episode), end="")
            break
        if i_episode == num_episodes: print('\n')
    return avg_rewards, best_avg_reward

def interact1(env, agent, num_episodes=20000, window=100):
    """ Monitor agent's performance.
    
    Params
    ======
    - env: instance of OpenAI Gym's Taxi-v1 environment
    - agent: instance of class Agent (see Agent.py for details)
    - num_episodes: number of episodes of agent-environment interaction
    - window: number of episodes to consider when calculating average rewards

    Returns
    =======
    - avg_rewards: deque containing average rewards
    - best_avg_reward: largest value in the avg_rewards deque
    """
    # initialize average rewards
    avg_rewards = deque(maxlen=num_episodes)
    # initialize best average reward
    best_avg_reward = -math.inf
    avg_reward = -math.inf
    # initialize monitor for most recent rewards
    samp_rewards = deque(maxlen=window)
    memory = Memory(max_size=20)
    batch_sample = 5
    step_total = 0
    
    # for each episode
    for i_episode1 in range(1, num_episodes+1):
        # begin the episode
        state = env.reset()
        step = 0
        # initialize the sampled reward
        samp_reward = 0
        #while True: #step <= 100
        while step <= 1000:
            step_total += 1
            step += 1
            
            if QNetwork == True:
                state_encode = to_categorical(state, num_classes=env.observation_space.n)
            else:
                state_encode = state
            #print(state_encode)
            # agent selects an action
            
            action1 = agent.select_action(state_encode,0,i_episode1)
            #action2 = agent2.select_action(state_encode,i_episode)
            #action3 = agent3.select_action(state_encode,i_episode)
            #action4 = agent4.select_action(state_encode,i_episode)
            #print(action1)
            action_all = action1
            #print(action_all)
            # agent performs the selected action
            next_state, reward, done, _ = env.step(action_all)
			# agent performs internal updates based on sampled experience
            ### Train using this data
            """
            if done:
                next_state = None
            """  
            if QNetwork == True:
                next_state_encode = to_categorical(next_state, num_classes=env.observation_space.n)
            else:
                next_state_encode = next_state  
            
            #memory.add((state_encode, action1, reward, next_state_encode, done))
            #print(next_state_encode)
           
            agent.step(state_encode, action1,0, reward, next_state_encode, done, i_episode1)
            #agent2.step(state_encode, action2, reward, next_state_encode, done, i_episode)
            #agent3.step(state_encode, action3, reward, next_state_encode, done, i_episode)
            #agent4.step(state_encode, action4, reward, next_state_encode, done, i_episode)
            
            
            #env.render()
            #print(action)
            #time.sleep(0.5)
            
            #print(step)
            """
            batch = memory.sample(1)
            #print(batch[0][0])
            state1 = batch[0][0]
            action1 = batch[0][1]
            reward1 = batch[0][2]
            next_state1 = batch[0][3]
            done1 = batch[0][4]
                
            agent.step(state1, action1, reward1, next_state1, done1, i_episode)
            """
            """"
            #env.render()
            batch_sample = 5
            if step % (batch_sample) == 0:
                if memory.count >= batch_sample:
                    batch = memory.sample(batch_sample)  
                    for i in range(len(batch)):
                        state1 = batch[i][0]
                        action1 = batch[i][1]
                        reward1 = batch[i][2]
                        next_state1 = batch[i][3]
                        done1 = batch[i][4]
                        agent.step(state1, action1,0, reward1, next_state1, done1, i_episode)            
                else:
                    batch = memory.sample(1)
                    state1 = batch[0][0]
                    action1 = batch[0][1]
                    reward1 = batch[0][2]
                    next_state1 = batch[0][3]
                    done1 = batch[0][4]
                    agent.step(state1, action1, reward1, next_state1, done1, i_episode)
            """
              
            """
            if memory.count >= batch_sample:
                batch = memory.sample(batch_sample)
                states = np.array([each[0] for each in batch])
                actions = np.array([each[1] for each in batch])
                rewards = np.array([each[2] for each in batch])
                next_states = np.array([each[3] for each in batch])
                agent.step(states, actions, rewards, next_states, done, i_episode)
            else:
                batch = memory.sample(1)
                agent.step(state, action, reward, next_state, done, i_episode)
              """
            # update the sampled reward
            samp_reward += reward
            # update the state (s <- s') to next time step
            state = next_state
            if done:
                #sampled reward
                
                samp_rewards.append(samp_reward)
                env.reset()
                state, reward, done, _ = env.step(env.action_space.sample())
                break
            else:
                state = next_state

                
        if (i_episode1 >= 100):
            # get average reward from last 100 episodes
            avg_reward = np.mean(samp_rewards)
            # append to deque
            avg_rewards.append(avg_reward)
            # update best average reward
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                """
        if (i_episode1%100 == 0):
            env.render()
            """
        # monitor progress
        print("\rEpisode {}/{} || Best average reward {} || average reward {} || episode reward {}".format(i_episode1, num_episodes, best_avg_reward, avg_reward, samp_reward), end="")
        sys.stdout.flush()
        # check if task is solved (according to OpenAI Gym)
        if best_avg_reward >= 9.7:
            print('\nEnvironment solved in {} episodes.'.format(i_episode1), end="")
            break
        if i_episode1 == num_episodes: print('\n')
    return avg_rewards, best_avg_reward

def decode(a1,a2,a3,a4):
    temp = 0
    temp = a1*8+a2*4+a3*2+a4*1
    return temp