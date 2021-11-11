#!/usr/bin/env python3
"""
play.py
Script that Loads weights to display a game played by the agent trained by train.py:
* Load the policy network saved in policy.h5
"""
import gym
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy
import keras as K
from keras.optimizers import Adam
AtariProcessor = __import__('train').AtariProcessor
create_q_model = __import__('train').create_q_model


if __name__ == '__main__':
    env = gym.make('Breakout-v0')
    env.reset()
    n_actions = env.action_space.n
    window = 4
    model = create_q_model(window, n_actions)
    memory = SequentialMemory(limit=1000000, window_length=window)
    processor = AtariProcessor()

    dqn = DQNAgent(model=model, nb_actions=n_actions,
                   policy=GreedyQPolicy(),
                   processor=processor, memory=memory)

    dqn.compile(Adam(lr=0.00025), metrics=['mae'])
    dqn.load_weights('policy.h5')
    dqn.test(env, nb_episodes=10, visualize=True)
