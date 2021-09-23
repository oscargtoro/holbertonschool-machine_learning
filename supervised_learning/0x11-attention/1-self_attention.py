#!/usr/bin/env python3
"""Module for the class SelfAttention
"""

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """Calculates the attention for machine translation based on
    https://arxiv.org/pdf/1409.0473.pdf
    """
    def __init__(self, units):
        """Class constructor.

        Args.
            units: Integer representing the number of hidden units in the
            alignment model.
        """
        super().__init__()
        self.W = tf.keras.layers.Dense(units=units)
        self.U = tf.keras.layers.Dense(units=units)
        self.V = tf.keras.layers.Dense(units=1)

    def call(self, s_prev, hidden_states):
        """

        Args.
            s_prev: tensor of shape (batch, units) containing the previous.
            decoder hidden state.
            hidden_states: tensor of shape (batch, input_seq_len, units)
            containing the outputs of the encoder.

        Returns.
            A tensor of shape (batch, units) that contains the context vector
            for the decoder and a tensor of shape (batch, input_seq_len, 1)
            that contains the attention weights.
        """

        exp_s_prev = tf.expand_dims(s_prev, axis=1)
        score = self.V(tf.nn.tanh(self.W(exp_s_prev) + self.U(hidden_states)))
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * hidden_states, axis=1)

        return context, weights
