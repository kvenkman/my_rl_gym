import numpy as np
from scipy import stats
import random
import matplotlib.pyplot as plt

# Calculating average rewards over 10 plays
def reward(prob):
    reward = 0
    for i in range(10):
        if random.random() < prob:
            reward += 1
    
    return reward
    
# Greedy method to select best arm based on memory array
def bestArm(a):
    bestArm = 0 # default to zero
    meanReward = 0
    #print("This is a: ", a)
    for key in a.keys():
        expected_reward = np.mean(a[key])
        if (expected_reward > meanReward):
            bestArm = key
            meanReward = expected_reward
                    
    return bestArm

n = 10 # for an n-armed bandit problem
arms = np.random.rand(n) # probability of payouts for each arm
epsilon = 0.01
nplays = int(1000)
history = np.zeros((nplays, 2)) # To keep track of history of choices
# Memory array
av = {}

for i in range(nplays):
    # Choosing the best action, exploit or explore
    if random.random() > epsilon: # Greedy selection
        choice = bestArm(av)
    else : # Random action
        choice = np.random.choice(range(10))

    # Reward obtained for said action
    current_reward = reward(arms[choice])
    
    # Updating memory for future calculations
    if not choice in av:
        av[choice] =  [current_reward]
    else:
        rewards = av[choice]
        rewards.append(current_reward)
        av[choice] = rewards

    history[i, :] = [choice, current_reward]

    runningMean = np.mean(history[:i+1, 1])
    plt.scatter(i, runningMean)

# print(av)

plt.xlabel("Plays")
plt.ylabel("Avg Reward")
plt.show()
print(arms, np.max(arms), np.where(arms == np.max(arms)))
plt.scatter(range(len(history[:, 0])), history[:, 0])   
plt.show()
