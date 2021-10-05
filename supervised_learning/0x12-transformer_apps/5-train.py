#!/usr/bin/env python3
"""Module for the methods that create and train a transformer model.
"""

import tensorflow as tf

Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    CustomSchedule class.
    """
    def __init__(self, d_model, warmup_steps=4000):
        """
        Class constructor.
        """
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        """
        call function.
        """
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def loss_function(loss_object, real, pred):
    """Applies padding mask to a calculated loss.

    Args:
        loss_object: sparse categorical crossentropy loss object.
        label: Actual label.
        prediction: Model prediction.

    Returns:
        tf.Tensor -- The masked loss
    """

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def train_step(transformer, optimizer, loss_object, train_loss, train_accuracy,
               inp, tar):
    """
    """

    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_mask, combined_mask, dec_mask = create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions = transformer(inp, tar_inp, True, enc_mask, combined_mask,
                                  dec_mask)
        loss = loss_function(loss_object, tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(tar_real, predictions)


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """creates and trains a transformer model.

    Args.
        N: Number of blocks in the encoder and decoder.
        dm: Dimensionality of the model.
        h: Number of heads.
        hidden: Number of hidden units in the fully connected layers.
        max_len: Maximum number of tokens per sequence.
        batch_size: Batch size for training.
        epochs: Number of epochs to train for.

    Returns.
        A trained model for machine translation of Portuguese to English.
    """

    Adam = tf.keras.optimizers.Adam
    scc = tf.keras.losses.SparseCategoricalCrossentropy
    sca = tf.keras.metrics.SparseCategoricalAccuracy
    data = Dataset(batch_size, max_len)

    learning_rate = CustomSchedule(dm)

    optimizer = Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    # sparse categorical crossentropy
    loss_object = scc(from_logits=True, reduction='none')
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = sca(name='train_accuracy')

    input_vocab_size = data.tokenizer_pt.vocab_size + 2
    target_vocab_size = data.tokenizer_en.vocab_size + 2

    transformer = Transformer(N=N,
                              dm=dm,
                              h=h,
                              hidden=hidden,
                              input_vocab=input_vocab_size,
                              target_vocab=target_vocab_size,
                              max_seq_input=max_len,
                              max_seq_target=max_len)

    # training
    for epoch in range(epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()

        for (batch, (inp, tar)) in enumerate(data.data_train):
            train_step(transformer, optimizer, loss_object, train_loss,
                       train_accuracy, inp, tar)

            if batch % 50 == 0:
                loss = train_loss.result()
                acc = train_accuracy.result()
                print('Epoch {}, batch {}: loss {} accuracy {}'.format(
                    epoch + 1, batch, loss, acc))

        loss = train_loss.result()
        acc = train_accuracy.result()
        print('Epoch {}: loss {} accuracy {}'.format(epoch + 1, loss, acc))

    return transformer
