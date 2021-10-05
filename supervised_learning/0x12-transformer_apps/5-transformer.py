#!/usr/bin/env python3
"""Module for the class transformer
"""

import tensorflow as tf
import numpy as np


def positional_encoding(max_seq_len, dm):
    """Calculates the positional encoding for a transformer.

    Args.
        max_seq_len: Integer representing the maximum sequence length.
        dm: the model depth.

    Returns.
        A numpy.ndarray of shape (max_seq_len, dm) containing the positional
        encoding vectors.
    """

    position = np.arange(max_seq_len)
    angle_rates = 1 / np.power(
        10000, (2 * (np.arange(dm)[np.newaxis, :] // 2)) / np.float32(dm))
    PE = position[:, np.newaxis] * angle_rates
    PE[:, 0::2] = np.sin(PE[:, 0::2])
    PE[:, 1::2] = np.cos(PE[:, 1::2])

    return PE


def sdp_attention(Q, K, V, mask=None):
    """calculates the scaled dot product attention.

    Args.
        Q: tensor with its last two dimensions as (..., seq_len_q, dk)
        containing the query matrix.
        K: tensor with its last two dimensions as (..., seq_len_v, dk)
        containing the key matrix.
        V: tensor with its last two dimensions as (..., seq_len_v, dv)
        containing the value matrix.
        mask: tensor that can be broadcast into (..., seq_len_q, seq_len_v)
        containing the optional mask, or defaulted to None.

    Returns.
        A tensor with its last two dimensions as (..., seq_len_q, dv)
        containing the scaled dot product attention and a tensor with its last
        two dimensions as (..., seq_len_q, seq_len_v) containing the attention
        weights.
    """

    matmul_QK = tf.matmul(Q, K, transpose_b=True)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = matmul_QK / tf.math.sqrt(dk)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(weights, V)

    return output, weights


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
        Q = self.split_heads(Q, batch)
        K = self.split_heads(K, batch)
        V = self.split_heads(V, batch)
        output, weights = sdp_attention(Q, K, V, mask)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch, -1, self.dm))
        output = self.linear(output)

        return output, weights

    def split_heads(self, x, batch):
        """Splits into heads.

        Args.
            x: tensor of shape (batch, seq_len_q, dk)

        Returns.
            A splited tensor.
        """

        return tf.transpose(tf.reshape(x, (batch, -1, self.h, self.depth)),
                            perm=[0, 2, 1, 3])


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


class Decoder(tf.keras.layers.Layer):
    """Decoder for a transformer
    """
    def __init__(self,
                 N,
                 dm,
                 h,
                 hidden,
                 target_vocab,
                 max_seq_len,
                 drop_rate=0.1):
        """Class constructor.

        Args.
            N: The number of blocks in the encoder.
            dm: The dimensionality of the model.
            h: The number of heads.
            hidden: The number of hidden units in the fully connected layer.
            target_vocab: The size of the target vocabulary.
            max_seq_len: The maximum sequence length possible.
            drop_rate: The dropout rate.
        """
        super().__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, self.dm)
        self.positional_encoding = positional_encoding(max_seq_len, self.dm)
        self.blocks = [
            DecoderBlock(self.dm, h, hidden, drop_rate) for _ in range(self.N)
        ]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Args.
            x: tensor of shape (batch, target_seq_len, dm)containing the input
            to the decoder.
            encoder_output: tensor of shape (batch, input_seq_len, dm)
            containing the output of the encoder.
            training: boolean to determine if the model is training.
            look_ahead_mask: The mask to be applied to the first multi head
            attention layer.
            padding_mask: The mask to be applied to the second multi head
            attention layer.

        Returns.
            A tensor of shape (batch, target_seq_len, dm) containing the
            decoder output.
        """
        target_seq_len = x.shape[1]
        output = self.embedding(x)
        output *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        output += self.positional_encoding[:target_seq_len]
        output = self.dropout(output, training=training)

        for i in range(self.N):
            output = self.blocks[i](output, encoder_output, training,
                                    look_ahead_mask, padding_mask)

        return output


class EncoderBlock(tf.keras.layers.Layer):
    """Creates an encoder block for a transformer.
    """
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Class constructor.

        Args.
            dm: The dimensionality of the model.
            h: The number of heads.
            hidden: The number of hidden units in the fully connected layer.
            drop_rate: The dropout rate.
        """
        super().__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(units=hidden,
                                                  activation='relu')
        self.dense_output = tf.keras.layers.Dense(units=dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate=drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, x, training, mask=None):
        """
        Args.
            x: tensor of shape (batch, input_seq_len, dm)containing the input
            to the encoder block.
            training: boolean to determine if the model is training.
            mask: The mask to be applied for multi head attention.

        Returns.
            A tensor of shape (batch, input_seq_len, dm) containing the block’s
            output.
        """
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.dense_hidden(out1)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)

        return self.layernorm2(out1 + ffn_output)


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
