from src.MiniPro.agent import Agent
from src.MiniPro.monitor import interact
import gym
def run():
    env = gym.make('Taxi-v3')
    agent = Agent()
    avg_rewards, best_avg_reward = interact(env, agent)