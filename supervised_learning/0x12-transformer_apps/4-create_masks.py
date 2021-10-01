#!/usr/bin/env python3
"""Module for the method create_masks
"""
import tensorflow.compat.v2 as tf


def create_masks(inputs, target):
    """Creates all masks for training/validation.

    Args.
        inputs: tf.Tensor of shape (batch_size, seq_len_in) that contains the
        input sentence
        target: tf.Tensor of shape (batch_size, seq_len_out) that contains the
        target sentence

    Returns.
        A tf.Tensor padding mask of shape (batch_size, 1, 1, seq_len_in),
        a tf.Tensor of shape (batch_size, 1, seq_len_out, seq_len_out) and a
        tf.Tensor padding mask of shape (batch_size, 1, 1, seq_len_in).
    """

    seq = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = seq[:, tf.newaxis, tf.newaxis, :]

    seq = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    decoder_mask = seq[:, tf.newaxis, tf.newaxis, :]
    mask_size = tf.shape(target)[1]
    la_mask = 1 - tf.linalg.band_part(tf.ones((mask_size, mask_size)), -1, 0)
    seq = tf.cast(tf.math.equal(target, 0), tf.float32)
    dec_target_padding_mask = seq[:, tf.newaxis, tf.newaxis, :]
    combined_mask = tf.maximum(dec_target_padding_mask, la_mask)

    return encoder_mask, combined_mask, decoder_mask
