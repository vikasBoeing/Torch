'''
this file will explains about the monte karlo method
'''
import gym
import sys
import numpy as np
from collections import defaultdict
from src.drl.plot_util import *
def run():
    '''
    entry function for the module
    '''
    # mc = MC()
    # mc.start()
    # # mc.randomPolicy()
    # Q = mc.mc_prediction_q(1, mc.generate_episode_from_limut_stochastic, 1)
    # print(Q)
    # # obtain the corresponding state-value function
    # V_to_plot = dict((k, (k[0] > 18) * (np.dot([0.8, 0.2], v)) + (k[0] <= 18) * (np.dot([0.2, 0.8], v))) \
    #                  for k, v in Q.items())
    #
    # # plot the state-value function
    # plot_blackjack_values(V_to_plot)
    mcAlpha = MCAlpha()
    policy, Q  = mcAlpha.MCControl(0.02)
    print(policy)
    print(Q)
class MC():
    def __init__(self):
        '''
        set the environment
        '''
        self.env = gym.make('Blackjack-v0')
        self.episodes = 1
        pass
    def start(self):
        print("Starting the application...!")
        print(self.env.action_space)
        print(self.env.observation_space)
        for i in range(3):
            print(self.generate_episode_from_limut_stochastic())
    def randomPolicy(self):
        for i_episode in range(self.episodes):
            state = self.env.reset()
            while True:
                print(state)
                action = self.env.action_space.sample()
                state, reward, done, info = self.env.step(action)
                if done:
                    print('End game! Reward: ', reward)
                    print('you won :)') if reward > 0 else print("you loose")
                    break
    def generate_episode_from_limut_stochastic(self):
        episode = []
        state = self.env.reset()
        while True:
            prob = [0.8, 0.2] if state[0] > 18 else [0.2, 0.8]
            action = np.random.choice(np.arange(2), p=prob)
            next_state, reward, done, info = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
            if done:
                break
        return episode

    def mc_prediction_q(self, nEpisodes, generateEpisode, gamma=1.0):

        # initialize empty dictionary of arrays
        returns_sum = defaultdict(lambda : np.zeros(self.env.action_space.n))
        N = defaultdict(lambda : np.zeros(self.env.action_space.n))
        Q = defaultdict(lambda : np.zeros(self.env.action_space.n))

        # loop over episodes
        for iEpisode in range(1, nEpisodes+1):
            if iEpisode % 100 == 0:
                print('\rEpisode {}/{}'.format(iEpisode, nEpisodes))
                sys.stdout.flush()
            episodes = generateEpisode()
            # obtain the state action reward
            states, actions, rewards = zip(*episodes)
            print("States : {}".format(states))
            print("Actions : {}".format(actions))
            print("Rewards : {}".format(rewards))
            # prepare for discounting
            discount = np.array([gamma**i for i in range(len(rewards)+1)])
            # update the sum of returns, number of visits, and action-value
            for i, state in enumerate(states):
                print("\rstate: {}".format(state))
                returns_sum[state][actions[i]] += sum(rewards[i: ] * discount[: -(1+i)])
                N[state][actions[i]] += 1.0
                Q[state][actions[i]] = returns_sum[state][actions[i]] / N[state][actions[i]]
        return Q


class MCAlpha():
    def __init__(self):
        self.env = gym.make('Blackjack-v0')
        self.episodes = 10000
        pass

    def generateEpisodesFromQ(self, Q, epsilon, nA):
        ''' generate an episode by following epsilon -greedy policy'''
        episode = []
        state = self.env.reset()
        while True:
            action = np.random.choice(np.arange(nA),
                                      p = self.getProb(Q[state], epsilon, nA)) if state in Q else self.env.action_space.sample()
            nextState, reward, done, info = self.env.step(action)
            episode.append((state, action, reward))
            state = nextState
            if done:
                break
        return episode

    def getProb(self, Q_s, epsilon, nA):
        '''obtain the action probabilities corresponding to epsillon-greedy policy'''
        policy_s = np.ones(nA) * epsilon /nA
        best_a = np.argmax(Q_s)
        policy_s[best_a] = 1 - epsilon + (epsilon / nA)
        return policy_s

    def update_Q(self, episode, Q, alpha, gamma):
        """
        update the action-value function estimating using the most recent episode
        """
        states, actions, rewards = zip(*episode)

        #3 prepair for disccounting
        discount = np.array([gamma**i for i in range(len(rewards)+ 1)])
        for i, state in enumerate(states):
            oldQ = Q[state][actions[i]]
            Q[state][actions[i]] = oldQ + alpha * (sum(rewards[i: ]* discount[:-(1+i)]) - oldQ)

        return Q

    def MCControl(self, alpha, gamma=1.0, eps_start=1.0, eps_decay=.99999, eps_min=0.05):
        nA = self.env.action_space.n

        # initialize empty dictionary of arrays
        Q = defaultdict(lambda : np.zeros(nA))
        epsilon = eps_start

        ## loop over episodes
        for i_episode in range(1, self.episodes+1):
            # monitor progress
            if i_episode % 1000 == 0:
                print("\rEpisode {}/{}.".format(i_episode, self.episodes), end="")
                sys.stdout.flush()
            # set the value of epsilon
            epsilon = max(epsilon * eps_decay, eps_min)
            # generate an episode by following epsilon-greedy policy
            episode = self.generateEpisodesFromQ( Q, epsilon, nA)
            # update the action-value function estimate using the episode
            Q = self.update_Q( episode, Q, alpha, gamma)
            # determine the policy corresponding to the final action-value function estimate
        policy = dict((k, np.argmax(v)) for k, v in Q.items())
        return policy, Q


