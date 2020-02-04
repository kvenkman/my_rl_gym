#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import gym
import random
import matplotlib.pyplot as plt
import time
from IPython.display import clear_output
import pickle
from collections import deque
import os


# In[2]:


# Import ML libraries
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model


# In[49]:


class Agent:
    def __init__(self, env):
        
        # Initializing Agent/Sim parameters
        # This agent will not have a q_table or an alpha value, 
        # since these will be configured in the DNN
        
        self.env = env
        
        self.gamma = 0.96 # discount factor
        self.epsilon = .9 # exploration rate
        self.epsilon_decay = 0.95
        self.epsilon_min = 0.01 
        
        self.alpha = 0.9
        self.alpha_decay = 0.01
        self.action = None
        
        self.n_observation_space = env.observation_space.n
        self.n_action_space = env.action_space.n
        
        self.current_state = env.reset()
        
        # To allow the agent to store (state, action) pairs which are
        # fed to the network for learning
        
        self.memory = deque(maxlen = 500)
        
        # Let's define the network in a separate method
        self.model = self.buildDQN()        

    def buildDQN(self):
        # Defining the DNN
        model = Sequential()
        model.add(Embedding(500, 10, input_length=1))
        model.add(Reshape((10,)))        
        model.add(Dense(24, input_dim=self.n_observation_space, activation='tanh'))
        model.add(Dense(24, activation='tanh'))
        model.add(Dense(self.n_action_space, activation='linear'))
        model.compile(loss='mse',optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))        
        
        return model
    
    def loadModel(self, filepath):
        self.model = load_model(filepath)
        
    def selectAction(self, state):
        if(random.uniform(0, 1) < self.epsilon):
            return env.action_space.sample()
        
        return np.argmax(self.model.predict(np.array(state).reshape(1)))

    def train(self, batch_size):
        
        if(len(self.memory) < batch_size):
            return
        
        x_batch, y_batch = [], []
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            y_target = self.model.predict(state)
            y_target[0][action] = reward if done else reward + \
                                    self.gamma*np.max(self.model.predict(next_state)[0])
            x_batch.append(state[0])
            y_batch.append(y_target[0])
            
        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
            
    def add_to_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def updateParameters(self):        
        self.epsilon = self.epsilon*self.epsilon_decay if self.epsilon > self.epsilon_min else self.epsilon_min        
    
    def reset(self):
        self.__init__(self.env)


env = gym.make("Taxi-v3")

dqnTaxiDriver = Agent(env)
#dqnTaxiDriver.reset()
dqnTaxiDriver.model = load_model("pickled_dqn.h5")

state = env.reset()

# Simulation parameters
batch_size = 32
num_of_episodes = 500 # 500
timesteps_per_episode = 200

env.reset()
for e in range(num_of_episodes):
    # Reset the enviroment
    state = env.reset()
    
    # Initialize variables
    reward = 0
    done = False
    print("Current episode number: ", e)
    #clear_output(wait=True)

    for timestep in range(timesteps_per_episode):
        # Run Action
        action = dqnTaxiDriver.selectAction(state)
        
        # Take action    
        next_state, reward, done, info = env.step(action) 
        dqnTaxiDriver.add_to_memory(np.array(state).reshape(1), action, reward, \
                                    np.array(next_state).reshape(1), done)
        
        state = next_state
        
        #env.render()
        
        dqnTaxiDriver.train(batch_size)
        
        if done:
            break
            
    if(e % 100 == 0):
        dqnTaxiDriver.updateParameters()
        dqnTaxiDriver.model.save("pickled_dqn.h5")

os.system("git add pickled_dqn.h5")
os.system("git commit -m \"autoupdate saved model\"")
os.system("git push origin master")        


# In[ ]:


#dqnTaxiDriver.model.save("pickled_dqn.hd5")
#os.system("git add pickled_dqn.h5")
#os.system("git commit -m \"autoupdate saved model\"")
#os.system("git push origin master")  

