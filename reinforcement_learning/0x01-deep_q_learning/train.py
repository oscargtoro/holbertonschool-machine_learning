#!/usr/bin/env python3
'''
Module for the script that handles the training for an agent to play atari's
breakout.
'''

import gym
import numpy as np
from PIL import Image
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.processors import Processor
import keras as K
from keras import layers
from keras.models import Model
from keras.optimizers import Adam


class AtariProcessor(Processor):
    '''
    The environment in which the game will be played.
    Processor for Atari.
    Prepocesses data based on Deep Learning
    Quick Reference by Mike Bernico.
    '''
    def process_observation(self, observation):
        '''
        loads image from array, resize to (84, 84) and converts to grayscale.

        Args.
            Observation: Array of Image to convert

        Returns.
            Array of converted image
        '''
        # Check if correct dimmensions or an actual RGB image
        if observation.ndim != 3:
            raise ValueError('Not an RGB image')
        # Load image from array
        img = Image.fromarray(observation)
        # Resize image and convert to grayscale
        img = img.resize((84, 84), Image.ANTIALIAS).convert('L')
        processed_observation = np.array(img)
        return processed_observation.astype('uint8')

    def process_reward(self, reward):
        '''
        Clips the rewards received during training to -1 and 1.

        Args.
            Reward: Reward received during training.
        '''
        return np.clip(reward, -1., 1.)


def create_q_model(window, n_actions):
    '''
    Creates a DQN model.

    Args.
        window: DQN inputs.
        n_actrions: DQN outputs.

    Returns.
        The DQN model.
    '''

    inputs = layers.Input(shape=(window, 84, 84))
    # rearranges layers
    dqn_layers = layers.Permute((2, 3, 1))(inputs)
    dqn_layers = layers.Conv2D(
        filters=16,
        activation='relu',
        kernel_size=8,
        strides=(4, 4),
        )(dqn_layers)
    dqn_layers = layers.Conv2D(
        filters=32,
        activation='relu',
        kernel_size=4,
        strides=(2, 2),
        )(dqn_layers)
    dqn_layers = layers.Conv2D(
        filters=64,
        activation='relu',
        kernel_size=2,
        )(dqn_layers)
    dqn_layers = layers.Flatten()(dqn_layers)
    dqn_layers = layers.Dense(512, activation="relu")(dqn_layers)
    dqn_layers = layers.Dense(256, activation="relu")(dqn_layers)
    outputs = layers.Dense(n_actions, activation="linear")(dqn_layers)
    return Model(inputs, outputs)


if __name__ == '__main__':
    env = gym.make('Breakout-v0')
    env.reset()
    n_actions = env.action_space.n
    window = 4
    model = create_q_model(window, n_actions)
    model.summary()
    memory = SequentialMemory(limit=1000000, window_length=window)
    processor = AtariProcessor()

    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr='eps',
        value_max=1.,
        value_min=.1,
        value_test=.05,
        nb_steps=1000000
        )

    dqn = DQNAgent(
        model=model,
        nb_actions=n_actions,
        policy=policy,
        memory=memory,
        processor=processor,
        nb_steps_warmup=50000,
        gamma=.99,
        target_model_update=10000,
        train_interval=4,
        delta_clip=1.
        )

    dqn.compile(Adam(lr=0.00025), metrics=['mae'])
    dqn.fit(env,
            nb_steps=10000000,
            log_interval=10000,
            visualize=False
            )

    dqn.save_weights('policy.h5', overwrite=True)
