#!/usr/bin/env python3
"""
Module for the Class DecoderBlock.
"""

import tensorflow as tf

MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """Creates a encoder block for a transformer.
    """
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Class constructor.

        Args.
            dm: Dimensionality of the model.
            h: Number of heads.
            hidden: Number of hidden units in the fully connected layer.
            drop_rate: The dropout rate.
        """
        super().__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(units=hidden,
                                                  activation='relu')
        self.dense_output = tf.keras.layers.Dense(units=dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate=drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(rate=drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Args.
            x: ensor of shape (batch, target_seq_len, dm)containing the input
            to the decoder block.
            encoder_output: tensor of shape (batch, input_seq_len, dm)
            containing the output of the encoder.
            training: boolean to determine if the model is training.
            look_ahead_mask: The mask to be applied to the first multi head
            attention layer.
            padding_mask: The mask to be applied to the second multi head
            attention layer.

        Returns.
            A tensor of shape (batch, target_seq_len, dm) containing the
            block’s output.
        """
        attn1, _ = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        attn2, _ = self.mha2(out1, encoder_output, encoder_output,
                             padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)
        ffn_output = self.dense_hidden(out2)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout3(ffn_output, training=training)

        return self.layernorm3(ffn_output + out2)
