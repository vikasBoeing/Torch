import numpy as np
from collections import defaultdict
import random

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.
        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = 0.01
        self.gamma = 1.0

    def select_action(self, state, eps):
        """ Given the state, select an action.
        Params
        ======
        - state: the current state of the environment
        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        ## want so select action with epsilon greedyli so thats why well use epsilon
        # with epsilon decay
        if random.random() > eps :
            return np.argmax(self.Q[state])
        else:
            return np.random.choice(self.nA)

    def step(self, state, action, reward, eps, next_state=None, next_action=None):
        """ Update the agent's knowledge, using the most recently sampled tuple.
        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        ## we will use the sarsa algorithm first
        current = self.Q[state][action]
        # Qsa_Next = self.Q[next_state][next_action] if next_state is not None else 0 ## SARSA

        Qsa_Next = np.max(self.Q[next_state]) if next_state is not None else 0 ## SARSA

        ### SARSA AVG
        # policy_s = np.ones(self.nA) * eps / self.nA
        # policy_s[np.argmax(self.Q[next_state])] = 1 - eps + (eps / self.nA)
        # Qsa_Next = np.dot(self.Q[next_state], policy_s)
        #
        target = reward + (self.gamma * Qsa_Next)
        newValue = current + (self.alpha * (target - current))
        self.Q[state][action] = newValue