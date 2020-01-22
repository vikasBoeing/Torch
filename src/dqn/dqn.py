import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from src.dqn.agent import Agent

class DQN():
    def __init__(self):
        '''
        constructor
        '''
        self.env = gym.make('LunarLander-v2')
        self.prt(self.env.action_space)
        self.prt(self.env.observation_space)
        self.agent = Agent(state_size=8, action_size=4, seed=0)
    pass

    def prt(self,*values):
        print("Values: {}".format(values))

    def start(self):
        agent = Agent(state_size=8, action_size=4, seed=0)
        state = self.env.reset()
        for j in range(200):
            action = agent.act(state)
            self.env.render()
            state, reward, done, _ = self.env.step(action)
            if done:
                print("Done")
                break
        self.env.close()

    def train(self, nEpisodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
        '''
        Deep Q Learning
        :param nEpisodes: maximum number of training episodes
        :param max_t: maximum number of timesteps per episode
        :param eps_start: starting value of epsilon, for epsilon-greedy action selection
        :param eps_end: minimum value of epsilon
        :param eps_decay: multiplicative factor
        :return: score
        '''
        scores = []
        scores_window =deque(maxlen=100)
        eps = eps_start
        for iEpisode in range(1, nEpisodes + 1):
            state = self.env.reset()
            score = 0
            for t in range(max_t):
                action = self.agent.act(state, eps)
                nextState, reard, done, _ = self.env.step(action)
                self.agent.step(state, action, reard, nextState, done)
                state = nextState
                score += reard
                if done:
                    break
            scores_window.append(score)
            scores.append(score)
            eps = max(eps_end, eps_decay*eps)
            # print('\r Episode {}\t Average Score: {:.2f}'.format(iEpisode, np.mean(scores_window)))
            if iEpisode % 100 == 0:
                print('\r Episode {}\tAverage Score: {:.2f}'.format(iEpisode, np.mean(scores_window)))
            if np.mean(scores_window) >= 200.0:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(iEpisode-100, np.mean(scores_window)))
                torch.save(self.agent.qnetwork_local.state_dict(), 'checkpoint.pth')
                break
        return scores

def run():
    '''
    starting point of the system
    :return:
    '''
    dnq = DQN()
    # dnq.start()
    scores = dnq.train()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
    pass