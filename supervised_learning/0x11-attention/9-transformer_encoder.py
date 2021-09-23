#!/usr/bin/env python3
"""Module for the class Encoder.
"""

import tensorflow as tf

positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """Encoder for a transformer.
    """
    def __init__(self,
                 N,
                 dm,
                 h,
                 hidden,
                 input_vocab,
                 max_seq_len,
                 drop_rate=0.1):
        """Class constructor.

        Args.
            N: The number of blocks in the encoder.
            dm: The dimensionality of the model.
            h: The number of heads.
            hidden: The number of hidden units in the fully connected layer.
            input_vocab: The size of the input vocabulary.
            max_seq_len: The maximum sequence length possible.
            drop_rate: The dropout rate.
        """
        super().__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, self.dm)
        self.positional_encoding = positional_encoding(max_seq_len, self.dm)
        self.blocks = [
            EncoderBlock(self.dm, h, hidden, drop_rate) for _ in range(N)
        ]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        Args.
            x: tensor of shape (batch, input_seq_len, dm)containing the input
            to the encoder.
            training: boolean to determine if the model is training.
            mask: The mask to be applied for multi head attention.

        Returns.
        A tensor of shape (batch, input_seq_len, dm) containing the encoder
        output
        """
        input_seq_len = x.shape[1]
        output = self.embedding(x)
        output *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        output += self.positional_encoding[:input_seq_len]
        output = self.dropout(output, training=training)

        for i in range(self.N):
            output = self.blocks[i](output, training, mask)

        return output
