import gym
import numpy as np
import math
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd

class CartPoleEnv():
    def __init__(self,buckets=(1, 1, 6, 12,), alpha=0.8, epsilon=0.6, decay_rate=0.99, gamma=1.0):
        self.env = gym.make('CartPole-v0')

        # parameters for tuning as episodes are run
        self.alpha = alpha 
        self.epsilon = epsilon
        self.decay_rate = decay_rate # for tuning alpha and epsilon during runs
        self.buckets = buckets
        self.gamma = gamma # discount factor, set to 1 as the goal is to survive as long as possible

        self.q_table = np.zeros(self.buckets + (self.env.action_space.n,))


    # To use Q learning on this problem, we need to discretize the continuous dimensions
    # Sourced from https://gist.github.com/muety/af0b8476ae4106ec098fea1dfe57f578
    def discretize(self, obs):
        upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2], math.radians(50)]
        lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2], -math.radians(50)]
        ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
        new_obs = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(self.buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
        return tuple(new_obs)

    def run(self, num_episodes):
        scores = deque(maxlen=100)

        for e in range(num_episodes):
            current_state = self.discretize(self.env.reset())

            self.alpha *= self.decay_rate
            self.epsilon *= self.decay_rate
            done = False
            steps = 0

            while not done:
                # self.env.render()
                do_explore = (1 - self.epsilon) <= np.random.uniform(0, 1)
                if do_explore:
                    # explore the action space
                    action = self.env.action_space.sample()
                else:
                    # choose the best action for the current state according to the q_table
                    action = np.argmax(self.q_table[current_state])

                obs, reward, done, _ = self.env.step(action)
                next_state = self.discretize(obs)
                
                old_qvalue = self.q_table[current_state][action]
                new_qvalue = (1 - self.alpha) * old_qvalue + self.alpha * (reward + self.gamma * np.max(self.q_table[next_state]))
                self.q_table[current_state][action] = new_qvalue
                current_state = next_state
                steps += 1

            scores.append(steps)
            mean_score = np.mean(scores)
            if mean_score >= 195 and e >= 100:
                print(f'Ran {e} episodes. Solved after {e - 100} trials')
                return scores
            if e % 100 == 0:
                print(f'Episode {e} - Average survival time over last 100 episodes was {mean_score} ticks.')

        print(f'No solution after {e} episodes')
        return scores

if __name__ == "__main__":
    env = CartPoleEnv()
    env.run(500)
