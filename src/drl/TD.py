import sys
import gym
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import numpy as np
import random
from src.drl.plot_util import plot_values

class TD():
    def __init__(self, envName):
        self.env = gym.make(str(envName))

    def sarsa(self, NEpisodes, alpha, gamma=1.0):
        ## Initialize action value function
        nA = self.env.action_space.n        # number of actions
        Q = defaultdict(lambda: np.zeros(nA))   # initialize emplty dict of array
        # monitor performance
        tmpScore = deque(maxlen=100)
        avgScore = deque(maxlen=NEpisodes)
        # initialize performance monitor
        # loop over episodes
        for iepisode in range(1, NEpisodes+1):
            # monitor progress
            if iepisode % 100 == 0:
                print("\r Episode {}/{}".format(iepisode, NEpisodes))
                sys.stdout.flush()
            score = 0
            state = self.env.reset()

            eps = 1.0 / iepisode
            action = self.epsilonGreedy(Q, state, nA, eps)      # epsilon greedy action selection

            while True:
                nextState, reward, done, info = self.env.step(action)

                score += reward
                if not done:
                    nextAction = self.epsilonGreedy(Q, nextState, nA, eps)
                    Q[state][action] = self.updateQsarsa(alpha, gamma, Q, state, action, reward, nextState, nextAction)
                    state = nextState
                    action = nextAction
                if done:
                    Q[state][action] = self.updateQsarsa(alpha, gamma, Q, state, action, reward)
                    tmpScore.append(score)
                    break
            if (iepisode % 100 == 0):
                avgScore.append(np.mean(tmpScore))
        # plot performance
        plt.plot(np.linspace(0, NEpisodes, len(avgScore), endpoint=False), np.asarray(avgScore))
        plt.xlabel('Episode Number')
        plt.ylabel('Average Reward (Over Next %d Episodes)' % 100)
        plt.show()
        return Q


    def updateQsarsa(self, alpha, gamma, Q, state, action, reward, nextState=None, nextAction=None):
        '''Return updated Q -values for the most recent experience'''
        current = Q[state][action]  # estimate in Q-table for current state and action pair

        # get the value of state, action pair at next time step
        # Qsa_next = Q[nextState][nextAction] if nextState is not None else 0  #SARSA
        Qsa_next = np.max(Q[nextState]) if nextState is not None else 0        # Q learning i.e. sarsaMax
        target = reward + (gamma * Qsa_next)
        newValue = current + (alpha * (target - current))
        return newValue
    def epsilonGreedy(self, Q, state, nA, eps):
        """
        select epsilon greedy action for supplied state
        """
        if random.random() > eps :
            return np.argmax(Q[state])
        else:
            return random.choice(np.arange(self.env.action_space.n))

def run():
    '''
    this function will run at start
    '''
    td = TD('CliffWalking-v0')
    td.sarsa(5000, 0.01)
