#!/usr/bin/env python3
"""Module for the class transformer
"""

import tensorflow as tf

Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.Model):
    """Creates a transformer network.
    """
    def __init__(self,
                 N,
                 dm,
                 h,
                 hidden,
                 input_vocab,
                 target_vocab,
                 max_seq_input,
                 max_seq_target,
                 drop_rate=0.1):
        """Class constructor.

        Args.
            N: The number of blocks in the encoder and decoder.
            dm: The dimensionality of the model.
            h: The number of heads.
            hidden: The number of hidden units in the fully connected layers.
            input_vocab: The size of the input vocabulary.
            target_vocab: The size of the target vocabulary.
            max_seq_input: The maximum sequence length possible for the input.
            max_seq_target: The maximum sequence length possible for the target
            drop_rate: The dropout rate.
        """
        super().__init__()
        self.encoder = Encoder(N, dm, h, hidden, input_vocab, max_seq_input,
                               drop_rate)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab, max_seq_target,
                               drop_rate)
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask, look_ahead_mask,
             decoder_mask):
        """
        Args.
            inputs: tensor of shape (batch, input_seq_len)containing the
            inputs.
            target: tensor of shape (batch, target_seq_len)containing the
            target.
            training: boolean to determine if the model is training.
            encoder_mask: The padding mask to be applied to the encoder.
            look_ahead_mask: The look ahead mask to be applied to the decoder.
            decoder_mask: The padding mask to be applied to the decoder.
        Returns.
            A tensor of shape (batch, target_seq_len, target_vocab) containing
            the transformer output.
        """
        enc_output = self.encoder(inputs, training, encoder_mask)
        dec_output = self.decoder(target, enc_output, training,
                                  look_ahead_mask, decoder_mask)

        final_output = self.linear(dec_output)
        return final_output
