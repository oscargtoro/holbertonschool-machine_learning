#!/usr/bin/env python3
"""Module for the class RNNEncoder
"""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """Calculates the attention for machine translation based on
    https://arxiv.org/pdf/1409.0473.pdf
    """
    def __init__(self, vocab, embedding, units, batch):
        """Class constructor.

        Args.
            vocab: Integer representing the size of the input vocabulary
            embedding: Integer representing the dimensionality of the embedding
            vector.
            units: Integer representing the number of hidden units in the RNN
            cell.
            batch: Integer representing the batch size.

        Returns.
            A tensor of shape (batch, units)containing the initialized hidden
            states.
        """

        super().__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)
        self.gru = tf.keras.layers.GRU(units=units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def initialize_hidden_state(self):
        """Initializes the hidden states for the RNN cell to a tensor of zeros.

        Returns.
            A tensor of shape (batch, units) containing the initialized hidden
            states.
        """

        hidden_state = tf.keras.initializers.Zeros()

        return hidden_state(shape=(self.batch, self.units))

    def call(self, x, initial):
        """Initializes the hidden states for the RNN cell to a tensor of zeros

        Args.
            x: tensor of shape (batch, input_seq_len) containing the input to
            the encoder layer as word indices within the vocabulary.
            initial: tensor of shape (batch, units) containing the initial
            hidden state.

        Returns.
            A tensor of shape (batch, input_seq_len, units)containing the
            outputs of the encoder and a tensor of shape (batch, units)
            containing the last hidden state of the encoder.
        """

        embeddings = self.embedding(x)
        outputs, hidden = self.gru(embeddings, initial_state=initial)

        return outputs, hidden
