#!/usr/bin/env python3
"""Module for the class MultiHeadAttention
"""

import tensorflow as tf

sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """Perform multi head attention.
    """
    def __init__(self, dm, h):
        """Class constructor.

        Args.
            dm: Integer representing the dimensionality of the model.
            h: Integer representing the number of heads.
        """
        super().__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(units=dm)
        self.Wk = tf.keras.layers.Dense(units=dm)
        self.Wv = tf.keras.layers.Dense(units=dm)
        self.linear = tf.keras.layers.Dense(units=dm)

    def call(self, Q, K, V, mask):
        """
        Args.
            Q: tensor of shape (batch, seq_len_q, dk) containing the input to
            generate the query matrix.
            K: tensor of shape (batch, seq_len_v, dk) containing the input to
            generate the key matrix.
            V: tensor of shape (batch, seq_len_v, dv) containing the input to
            generate the value matrix.
            mask: always None.

        Returns.
            A tensor with its last two dimensions as (..., seq_len_q, dm)
            containing the scaled dot product attention and a tensor with its
            last three dimensions as (..., h, seq_len_q, seq_len_v) containing
            the attention weights.
        """

        batch = tf.shape(Q)[0]
        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        scaled_attention, weights = sdp_attention(Q, K, V, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = \
            tf.reshape(scaled_attention, (batch, -1, self.dm))
        output = self.linear(concat_attention)

        return output, weights

    def split_heads(self, x):
        """Splits into heads.

        Args.
            x: tensor of shape (batch, seq_len_q, dk)

        Returns.
            A splited tensor.
        """

        batch = tf.shape(x)[0]
        return tf.transpose(tf.reshape(x, (batch, -1, self.h, self.depth)),
                            perm=[0, 2, 1, 3])
