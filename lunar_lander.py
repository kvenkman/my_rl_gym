from collections import deque
import random
import numpy as np
import gym
import os
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from IPython.display import clear_output
import matplotlib.pyplot as plt

# To disable GPU (e.g. while testing)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class Agent():
    def __init__(self, env, qnetwork_file=None, batch_size = 256, train = True, render = False):
        self.env = env
        self.epsilon = 1.0 if (train) else 0.
        self.gamma = 0.99
        self.batch_size = batch_size
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995 if (train) else 1.
        self.alpha = 0.0001
        self.memory = deque(maxlen=10000)

        self.train = train # Easy way switch train/test
        self.render = render

        # Qnetwork and target network
        self.qnetwork = self.defineNetwork() if (qnetwork_file == None) else load_model(qnetwork_file)
        self.target = self.defineNetwork()
        self.alignTarget()

    def defineNetwork(self):
        model = Sequential()
        model.add(Dense(32, input_dim = self.env.observation_space.shape[0], activation="relu"))  # input layer
        model.add(Dense(32, activation="relu")) # hidden layer
        model.add(Dense(self.env.action_space.n, activation='linear')) # output layer
        model.compile(loss='mse',optimizer=Adam(lr=self.alpha))
        return model

    def alignTarget(self):
        self.target.set_weights(self.qnetwork.get_weights())

    def setSimParameters(self, episodes=1000, ntimesteps=1000):
        self.episodes = episodes if (self.train) else 10
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

            train_states = []
            target_rewards = []

            for state, action, reward, next_state, done in batch:
                target = self.qnetwork.predict(state)
                if (done):
                    target[0][action] = reward
                else:
                    t = self.target.predict(next_state)
                    target[0][action] = reward + self.gamma*np.amax(t)

                train_states.append(state)
                target_rewards.append(target)

            self.qnetwork.fit(np.squeeze(train_states), np.squeeze(target_rewards), epochs = 1, verbose = 0)

    def updateEpsilon(self):
        self.epsilon = np.maximum(self.epsilon_decay*self.epsilon, self.epsilon_min)

# Define environment
env_name = "LunarLander-v2"
env = gym.make(env_name)

# Define agent, sim parameters
myLander = Agent(env, qnetwork_file="lander_qnetwork_2999_0.01_0.9995.h5", batch_size = 128, train = False, render = True) #
myLander.setSimParameters(episodes = 10000, ntimesteps = 2000)

save_point = 100
cumulative_rewards = [0]
episode_list = [0]
cumulative_reward = 0
for i in range(myLander.episodes): # myLander.episodes

    state = env.reset()
    clear_output(wait=True) # Works only inside jupyter notebooks
    cumulative_reward = 0

    for _ in range(myLander.ntimesteps): # myLander.ntimesteps
        if myLander.render:
            env.render()

        action = myLander.selectAction(state.reshape(1, 8))
        new_state, reward, done, _ = env.step(action)
        cumulative_reward += reward
        myLander.addToMemory(state.reshape(1, 8), action, reward, new_state.reshape(1, 8), done)
        state = new_state
        if (done):
            break

    print("Episode number: {} of {}. Last reward: {}".format(i+1, myLander.episodes, cumulative_reward))

    myLander.optimize()
    myLander.updateEpsilon()
    myLander.alignTarget()
    if (myLander.train and (i==save_point)):
        save_file_name = "lander_qnetwork_"+str(i)+".h5"
        myLander.qnetwork.save(save_file_name)
        save_point *= 2
        cumulative_rewards.append(cumulative_reward)
        episode_list.append(i)
#
# myLander.qnetwork.save("lander_qnetwork_"+str(save_point)+".h5")
# os.system("git add lander_*.h5")
# os.system("git commit -m \"autoupdate saved lunar lander agent network\"")
# os.system("git add rewards.png")
# os.system("git commit -m \"rewards plot\"")
# os.system("git push origin master")
if (myLander.train):
    plt.figure(figsize=(12,6))
    plt.plot(episode_list, cumulative_rewards)
    plt.title("Rewards vs Episode #", size=15)
    plt.xlabel("Episode #", size=12)
    plt.ylabel("Rewards", size=12)
    plt.grid()
    plt.savefig("rewards.png")
