import numpy as np
import gym
import matplotlib.pyplot as plt

# sourced from https://github.com/kvfrans/openai-cartpole/blob/master/cartpole-random.py
# found at http://kvfrans.com/simple-algoritms-for-solving-cartpole/

def run_episode(env, parameters):  
    observation = env.reset()
    totalreward = 0
    for _ in range(200):
        action = 0 if np.matmul(parameters,observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        totalreward += reward
        if done:
            break
    return totalreward

def train():
    env = gym.make('CartPole-v0')

    counter = 0
    bestparams = None
    bestreward = 0
    for _ in range(1000):
        counter += 1
        parameters = np.random.rand(4) * 2 - 1
        reward = run_episode(env,parameters)
        if reward > bestreward:
            bestreward = reward
            bestparams = parameters
            if reward == 200:
                break


    return counter

results = []
for _ in range(100):
    results.append(train())

plt.hist(results,50, facecolor='g', alpha=0.75)
plt.xlabel('Episodes required to reach 200')
plt.ylabel('Frequency')
plt.title('Histogram of Random Search')
plt.show()

print(np.sum(results) / 1000.0)