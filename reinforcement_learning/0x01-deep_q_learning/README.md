# Deep Q-Learning

Deep Q-Learning combines the concepts of Q-Learning and Depp Neural Networks to reduce the performance costs of only using Q-Learning.

The idea of this project is to apply the concepts of policy network, replay memory and target network while using keras-rl to train an agent to play atari's breakout.

# Usage & Requirements

This repository was created and coded using ubuntu xenial64(which comes with python 3.5.2) and anaconda 4.2.0 (latest version to include python 3.5). To test this repository is necessary to use a virtual environment using anaconda with the following requirements:

```
numpy==1.18.5
tensorflow==1.14
gym==0.17.2
keras==2.2.5
keras-rl==0.4.2
tensorflow==1.14
Pillow
h5py
atari-py
```

Any attempt to run without using a virtual environment with anaconda or using tensorflow >= 2.x ended in errors calling DNQAgent from keras-rl, The installation of python-opengl was also necessary to see the agent play the game.
