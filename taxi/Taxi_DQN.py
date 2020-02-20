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

# Import ML libraries
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import one_hot

# To disable GPU (e.g. while testing)
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class Agent():
    def __init__(self, env, qnetwork_file=None, batch_size = 32, alpha = 0.0001, epsilon_decay = 0.99941, train = True):
        self.env = env
        self.epsilon = 1.0 if (train) else 0.
        self.gamma = 0.99
        self.batch_size = batch_size
        self.epsilon_min = 0.05
        self.epsilon_decay = epsilon_decay
        self.alpha = 0.0001
        self.memory = deque(maxlen=100000)

        # To switch train/test
        self.train = train

        # Qnetwork and target network
        self.qnetwork = self.defineNetwork() if (qnetwork_file == None) else load_model(qnetwork_file)
        self.target = self.defineNetwork()
        self.alignTarget()

    def defineNetwork(self):
        model = Sequential()
        model.add(Embedding(500, 16, input_length=1))
        #model.add(Dense(16, input_dim=np.array(1), activation="linear"))
        model.add(Reshape((16,)))
        model.add(Dense(16, activation="relu"))  # hidden layer
        model.add(Dense(16, activation="relu")) # hidden layer
        model.add(Dense(self.env.action_space.n, activation='linear')) # output layer
        model.compile(loss='mse',optimizer=Adam(lr=self.alpha))
        return model

    def alignTarget(self):
        self.target.set_weights(self.qnetwork.get_weights())

    def setSimParameters(self, episodes=100, ntimesteps=200):
        self.episodes = episodes if (self.train) else 100
        self.ntimesteps = ntimesteps

    def selectAction(self, state):
        if(random.uniform(0, 1) < self.epsilon):
            return env.action_space.sample()
        return np.argmax(self.qnetwork.predict(np.array(state)))

    def addToMemory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def optimize(self): # , state, action, reward, next_state, done
        if (self.train and (len(self.memory) > self.batch_size)):
            batch = np.array(random.sample(self.memory, self.batch_size))

            train_states = np.zeros((self.batch_size, 1))
            target_rewards = np.zeros((self.batch_size, self.env.action_space.n))

            for i, (state, action, reward, next_state, done) in enumerate(batch):
                target = np.squeeze(self.qnetwork.predict(state))
                if (done):
                    target[action] = reward
                else:
                    t = self.target.predict(next_state)
                    target[action] = reward + self.gamma*np.amax(t)

                train_states[i] = state
                target_rewards[i, :] = target

            # print("Train_states, target rewards shape {} {}".format(train_states.shape, target_rewards.shape))
            self.qnetwork.fit(train_states, target_rewards, epochs = 1, verbose = 0)

    def updateEpsilon(self):
        self.epsilon = np.maximum(self.epsilon_decay*self.epsilon, self.epsilon_min)

env = gym.make("Taxi-v3")
alpha_space = [0.01] # [0.01, 0.001, 0.0001]
epsilon_d_space = [0.9995] # [0.9994, 0.9996, 0.99995]
n_episodes = 3000
cumulative_rewards = np.zeros((n_episodes, len(alpha_space), len(epsilon_d_space)))

for a, alpha in enumerate(alpha_space):
    for e, epsilon_d in enumerate(epsilon_d_space):
        # Define agent, sim parameters
        taxiAgent = Agent(env, qnetwork_file=None, batch_size = 64, alpha = alpha, epsilon_decay = epsilon_d, train = True)
        taxiAgent.setSimParameters(episodes = n_episodes, ntimesteps = 200)

        for i in range(1, taxiAgent.episodes+1):
            state = env.reset()
            clear_output(wait=True) # Works only inside jupyter notebooks

            for j in range(taxiAgent.ntimesteps): # taxiAgent.ntimesteps
                if not taxiAgent.train:
                    env.render()
                    print("Current episode number, timesteps, epsilon: ", i, taxiAgent.ntimesteps, j, taxiAgent.epsilon)
                    clear_output(wait=True)

                action = taxiAgent.selectAction(np.array(state).reshape(1))
                new_state, reward, done, _ = env.step(action)
                cumulative_rewards[i, a, e] += reward
                taxiAgent.addToMemory(np.array(state).reshape(1), action, reward, np.array(new_state).reshape(1), done)
                state = new_state

                if (done):
                    break

            # cumulative_rewards.append(cumulative_reward)
            # episode_list.append(i)
            taxiAgent.optimize()
            taxiAgent.updateEpsilon()
            taxiAgent.alignTarget()

            print("Episode number: {} of {}. Current alpha, epsilon: {}, {}. Last episode reward: {}".format(
                                                                         i, taxiAgent.episodes, alpha, taxiAgent.epsilon,
                                                                         cumulative_rewards[i, a, e]))
            # if (taxiAgent.train and (i == save_point)):
            #     taxiAgent.qnetwork.save("taxi_qnetwork_"+str(i)+".h5")
            #     os.system("git add taxi_*.h5")
            #     os.system("git commit -m \"autoupdate saved taxi agent network \"")
            #     save_point += 1000

        taxiAgent.qnetwork.save("taxi_qnetwork_"+str(alpha)+"_"+str(epsilon_d)+".h5")

plt.figure(figsize=(12,6))
for a in len(alpha_space):
  for ed in len(epsilon_decay_space):
    plt.plot(cumulative_rewards[:, a, ed], label="alpha: {}, epsilon_decay: {}".format(alpha_space[a], epsilon_decay_space[ed]))

plt.title("Rewards vs Episode #", size=15)
plt.xlabel("Episode #", size=12)
plt.ylabel("Rewards", size=12)
plt.grid()
plt.savefig("taxi_rewards.png")
os.system("git add taxi_rewards.png")
os.system("git commit -m \"Taxi rewards plot\"")
os.system("git push origin master")
