from collections import deque
import random
import numpy as np
import gym
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

class Agent():
    def __init__(self, env):
        self.env = env
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.gamma = 0.99
        self.batch_size = 32
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.alpha = 0.01
        self.memory = deque(maxlen=10000)

        self.train = True # Easy way switch train/test

        # Qnetwork and target network
        self.qnetwork = self.defineNetwork()
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

    def setSimParameters(self, episodes=100, ntimesteps=1000):
        self.episodes = episodes
        self.ntimesteps = ntimesteps

    def loadModel(self, qnetwork_path, target_path):
        self.qnetwork = load_model(qnetwork_path)
        self.target = load_model(target_path)

    def selectAction(self, state):
        if(random.uniform(0, 1) < self.epsilon):
            return env.action_space.sample()
        return np.argmax(self.qnetwork.predict(np.array(state)))

    def addToMemory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def optimize(self): # , state, action, reward, next_state, done
        if (self.train and (len(self.memory) > self.batch_size)):
            batch = np.array(random.sample(self.memory, self.batch_size))

            for state, action, reward, next_state, done in batch:
                target = self.qnetwork.predict(state)
                if (done):
                    target[0][action] = reward
                else:
                    t = self.target.predict(next_state)
                    target[0][action] = reward + self.gamma*np.amax(t)

                self.qnetwork.fit(state, target, epochs = 1, verbose = 0)

    def updateEpsilon(self):
        self.epsilon *= self.epsilon_decay


env_name = "LunarLander-v2"
env = gym.make(env_name)
myLander = Agent(env)

# Define simulation length
myLander.setSimParameters(episodes = 100, ntimesteps = 2000)

state = env.reset()
# To test lander
myLander.train = False
myLander.epsilon = 0
myLander.loadModel("pickled_lander_qnetwork.h5", "pickled_lander_target.h5")

for i in range(100): # myLander.episodes
    if myLander.train:
        print("Current episode number: ", i)
    current_reward = [0]

    for _ in range(myLander.ntimesteps): # myLander.ntimesteps
        if not myLander.train:
            env.render()

        action = myLander.selectAction(state.reshape(1, 8))
        new_state, reward, done, _ = env.step(action)

        current_reward.append(reward)

        myLander.addToMemory(state.reshape(1, 8), action, reward, new_state.reshape(1, 8), done)
        myLander.optimize()

        state = new_state

        if (done):
            break

    myLander.updateEpsilon()
    myLander.alignTarget()


myLander.qnetwork.save("pickled_lander_qnetwork.h5")
myLander.target.save("pickled_lander_target.h5")
