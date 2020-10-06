import gym
import numpy as np
import time, pickle, os

# Code originally sourced from https://medium.com/swlh/introduction-to-reinforcement-learning-coding-q-learning-part-3-9778366a41c0
# It has been rewritten to be object oriented as well as to implement decaying hyper paramters

def clear(): os.system('cls')

class QAgent:
    def __init__(self, env_name, total_episodes):
        self.env_name = env_name
        self.env = gym.make(self.env_name)
        self.epsilon = 0.9
        self.gamma = 0.99
        self.alpha = 0.8
        self.decay_rate = 0.999
        self.q_table = np.zeros([self.env.observation_space.n,self.env.action_space.n])
        self.max_steps = 100
        self.total_episodes = total_episodes

    def decay_hyper_params(self):
        if self.alpha > 0.1: self.alpha *= self.decay_rate
        if self.epsilon > 0.01: self.epsilon *= self.decay_rate

    def choose_action(self, state):
        action=0
        if np.random.uniform(0, 1) < self.epsilon:
            action =self.env.action_space.sample()
        else:
            action = np.argmax(self.q_table[state])

        return action

    def learn(self, state, next_state, reward, action):
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state, action] = new_value

    def run(self):
        for episode in range(self.total_episodes):
            state = self.env.reset()
            step_counter = 0
            if episode % 100 == 0: 
                clear()
                print(episode)

            while step_counter < self.max_steps:

                action = self.choose_action(state)  

                next_state, reward, done, info = self.env.step(action)  

                self.learn(state, next_state, reward, action)

                state = next_state

                step_counter += 1
            
                if done: break

            self.decay_hyper_params()

        self.write_to_pickle(f"{self.env_name}_{total_episodes}.pkl")


    def write_to_pickle(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self.q_table, f)

    def read_from_pickle(self, file_name):
        with open(file_name, 'rb') as f:
            self.q_table = pickle.load(f)

    def print_run(self, total_episodes, save_file, verbose=False):
        agent.read_from_pickle(save_file)
        total_wins, total_losses, total_timesteps = 0, 0, 0
        for episode in range(total_episodes):
            state =self.env.reset()
            
            done = False
            step_counter = 0

            while not done:
                step_counter +=1
                action = np.argmax(self.q_table[state])
                next_state, reward, done, info =self.env.step(action)  
                state = next_state

                if verbose:
                    clear()
                    self.env.render()
                    print(f"Episode: {episode}")
                    time.sleep(0.1)

            if reward > 0:
                total_wins +=1
            else:
                total_losses += 1

            total_timesteps += step_counter

        print(f"Out of {total_episodes} episodes\nThe agent won {total_wins}\nThe agent lost {total_losses}")
        print(f"Average timesteps per episode: {total_timesteps / total_episodes}")


if __name__ == '__main__':
    # You can choose which environment you would like to use by commenting/uncommenting one of these lines
    # env_name = 'Taxi-v3'
    env_name = 'FrozenLake-v0'

    # The number of episodes the agent should be trained for
    total_episodes = 10000

    agent = QAgent(env_name, total_episodes)
    
    # Train the agent for the specified number of episodes
    # After doing so, the generated Q table will be serialized and saved as {env_name}_{total_episodes}.pkl
    agent.run()


    # print_run will yield the results of an agent, following a Q-learning policy
    # Set verbose to true if you would like to have the environment rendered so that you can see it solve the problem
    # The filename must correspond to a serialized q_table, saved after executing the .run() function
    filename = f"{env_name}_{total_episodes}.pkl"
    agent.print_run(1000, filename, verbose=False)


