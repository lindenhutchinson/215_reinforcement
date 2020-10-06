import os
from time import sleep
import gym
import numpy as np
import random 
import os
import pandas as pd

def clear():
    os.system('cls')

class TaxiEnv:
    def __init__(self, env_name, q_table_csv):
        self.env = gym.make(env_name).env
        # initializing the q table full of zeros
        self.q_table = np.zeros([self.env.observation_space.n, self.env.action_space.n])

        # keeping this incase we want to save our q table
        self.q_table_csv = q_table_csv

        # # hyperparameters - these should all decrease
        self.alpha = 0.8
        self.epsilon = 0.5
        self.decay_rate = 0.99
        self.gamma = 0.6

    
    def read_q_table(self):
        file = pd.read_csv(self.q_table_csv, header=None)
        self.q_table = file.values

    def write_q_table(self):
        with open(self.q_table_csv, "w") as fn:
            for row in self.q_table:
                row_count = 0
                for item in row:
                    fn.write(f"{item}{',' if row_count != len(row)-1 else ''}")
                    row_count +=1

                fn.write("\n")
        
    def run_episodes(self, num_episodes):
        for i in range(1, num_episodes):
            if self.alpha > 0.1: self.alpha *= self.decay_rate
            if self.epsilon > 0.1: self.epsilon *= self.decay_rate
            if i % 100 == 0:
                clear()
                print(f"Episode: {i}")

            state=self.env.reset()
            epochs, penalties, reward = 0,0,0
            done = False
            while not done:
                if random.uniform(0,1) < self.epsilon:
                    # explore the action space
                    action = self.env.action_space.sample()
                else:
                    # choose the best action for the current state according to the q_table
                    action = np.argmax(self.q_table[state])

                next_state, reward, done, info = self.env.step(action)

                old_value = self.q_table[state, action]
                next_max = np.max(self.q_table[next_state])

                new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
                self.q_table[state, action] = new_value

                if reward == -10:
                    penalties +=1

                state = next_state
                epochs +=1

        self.write_q_table()
        print(f"Training finished.\nQ-table saved as {self.q_table_csv}")

    def run_random_episodes(self, num_episodes):
        total_epochs, total_penalties = 0, 0
        episode_frames = []
        for i in range(0, num_episodes):
            self.env.reset()  
            epochs = 0
            penalties, reward = 0, 0

            done = False

            while not done:
                action = self.env.action_space.sample()
                state, reward, done, info = self.env.step(action)

                if reward == -10:
                    penalties += 1

                epochs += 1
            total_penalties += penalties
            total_epochs += epochs

        print(f"Results after {num_episodes} episodes:")
        print(f"Average timesteps per episode: {total_epochs / num_episodes}")
        print(f"Average penalties per episode: {total_penalties / num_episodes}")


    def print_episodes(self, num_episodes):
        self.read_q_table()
        total_epochs, total_penalties = 0, 0
        episode_frames = []
        tries = 0
        for i in range(num_episodes):
            frames = []
            state = self.env.reset()
            epochs, penalties, reward = 0, 0, 0
            done = False
            
            while not done:
                action = np.argmax(self.q_table[state])
                state, reward, done, info = self.env.step(action)

                if reward == -10:
                    penalties += 1

                epochs += 1
                frames.append({
                    'frame': self.env.render(mode='ansi'),
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'episode':i
                    }
                )

                tries +=1
            episode_frames.append(frames)
            total_penalties += penalties
            total_epochs += epochs
           

        for frames in episode_frames:
            self.print_frames(frames)
            sleep(0.5)

        print(f"Results after {num_episodes} episodes:")
        print(f"Average timesteps per episode: {total_epochs / num_episodes}")
        print(f"Average penalties per episode: {total_penalties / num_episodes}")

    
    def print_frames(self, frames):
        for i, frame in enumerate(frames):
            clear()
            print(frame['frame'])
            print(f"Timestep: {i + 1}")
            print(f"State: {frame['state']}")
            print(f"Action: {frame['action']}")
            print(f"Reward: {frame['reward']}")
            print(f"Episode: {frame['episode']}")
            sleep(.1)



if __name__ == "__main__":
    e = TaxiEnv("Taxi-v3","q_table.csv")
    e.run_episodes(10000) # train the agent over 10000 episodes
    e.print_episodes(10) # print the results of the training
    # e.run_random_episodes(5) # attempt to solve the problem by using a random policy
    
