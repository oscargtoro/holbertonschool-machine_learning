#!/usr/bin/env python3
"""Module for the class RNNDecoder.
"""

import tensorflow as tf

SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """Decode for machine translation.
    """
    def __init__(self, vocab, embedding, units, batch):
        """Class constructor.

        Args.
            vocab: Integer representing the size of the output vocabulary.
            embedding: Integer representing the dimensionality of the embedding
            vector.
            units: Integer representing the number of hidden units in the RNN
            cell.
            batch: Integer representing the batch size.

        """
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)
        self.gru = tf.keras.layers.GRU(units=units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.F = tf.keras.layers.Dense(units=vocab)

    def call(self, x, s_prev, hidden_states):
        """
        Args.
            x: tensor of shape (batch, 1) containing the previous word in the
            target sequence as an index of the target vocabulary.
            s_prev: tensor of shape (batch, units) containing the previous
            decoder hidden state.
            hidden_states: tensor of shape (batch, input_seq_len, units)
            containing the outputs of the encoder.
        Returns.
            A tensor of shape (batch, vocab) containing the output word as a
            one hot vector in the target vocabulary and a tensor of shape
            (batch, units) containing the new decoder hidden state.
        """

        _, units = s_prev.shape
        attention = SelfAttention(units)
        context, _ = attention(s_prev, hidden_states)
        embeddings = self.embedding(x)
        exp_context = tf.expand_dims(context, axis=1)
        concat_input = tf.concat([exp_context, embeddings], axis=-1)
        outputs, s = self.gru(concat_input)
        outputs = tf.reshape(outputs, (outputs.shape[0], outputs.shape[2]))
        y = self.F(outputs)

        return y, s
