# tf_lib holds tensorflow related "plumbing" utility functions.

import numpy as np
import tensorflow as tf


MIN_POSSIBLE_VALUE = np.finfo(np.float32).min


def masked_reduce_max(t: tf.Tensor, mask: tf.Tensor, axis=None,
                      keepdims=None, name=None) -> tf.Tensor:
    with tf.name_scope(name=name or 'masked_reduce_max'):
        min_mask = tf.where(mask, t, tf.ones_like(t) * MIN_POSSIBLE_VALUE)
        return tf.math.reduce_max(min_mask, axis=axis, keepdims=keepdims)


def masked_softmax(logits: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    logits_max = masked_reduce_max(logits, mask, axis=-1, keepdims=1)
    logtis_max = tf.stop_gradient(logits_max)
    normalized_logits = logits - logits_max
    zeros = tf.zeros_like(logits)
    normalized_exp = tf.where(mask, tf.math.exp(normalized_logits), zeros)
    normalized_exp_sum = tf.reduce_sum(normalized_exp, axis=-1, keepdims=True)
    normalized_log_sum = tf.math.log(normalized_exp_sum)
    return tf.where(mask, normalized_exp / normalized_exp_sum, zeros)


def range_like(original: tf.Tensor) -> tf.Tensor:
    """Returns a tensor with same shape, but with indices values of the last dimension."""
    indices = tf.range(original.shape[-1])
    repeats = []
    for ii, dim in enumerate(original.shape):
        if ii != len(original.shape) - 1:
            repeats.append(dim)
            indices = tf.expand_dims(indices, axis=0)
        else:
            repeats.append(1)
    return tf.tile(indices, repeats)


def masked_first(values: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    """Returns the first element of the last dimension that is not masked."""
    indices = range_like(values)
    max_index = values.shape[-1] + 1
    indices = tf.where(mask, indices, max_index * tf.ones_like(indices))
    indices = tf.reduce_min(indices, axis=-1, keepdims=True)
    return tf.gather_nd(values, indices, batch_dims=len(indices.shape) - 1)


def masked_last(values: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    """Returns the first element of the last dimension that is not masked."""
    indices = range_like(values)
    indices = tf.where(mask, indices, -1 * tf.ones_like(indices))
    indices = tf.reduce_max(indices, axis=-1, keepdims=True)
    return tf.gather_nd(values, indices, batch_dims=len(indices.shape) - 1)
