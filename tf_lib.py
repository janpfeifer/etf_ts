# tf_lib holds tensorflow related "plumbing" utility functions.

from absl import logging
import numpy as np
import tensorflow as tf


MIN_POSSIBLE_VALUE = np.finfo(np.float32).min
MAX_POSSIBLE_VALUE = np.finfo(np.float32).max


def config_gpu():
    """Trivial GPU configuration: not all memory is needed for this."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            logging.info(f'{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs')
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            logging.info(e)


def masked_reduce_max(t: tf.Tensor, mask: tf.Tensor, default: float,
                      axis=None, keepdims=None, name=None) -> tf.Tensor:
    with tf.name_scope(name=name or 'masked_reduce_max'):
        min_mask = tf.where(mask, t, tf.ones_like(t) * MIN_POSSIBLE_VALUE)
        reduced = tf.math.reduce_max(min_mask, axis=axis, keepdims=keepdims)
        reduced_mask = tf.reduce_any(mask, axis=axis, keepdims=keepdims)
        return tf.where(reduced_mask, reduced, tf.ones_like(t) * default)


def masked_reduce_min(t: tf.Tensor, mask: tf.Tensor, default: float,
                      axis=None, keepdims=None, name=None) -> tf.Tensor:
    """Like tf.reduce_min, but numbers where mask=False don't participate. Completely masked values are replaced by default."""
    with tf.name_scope(name=name or 'masked_reduce_max'):
        max_mask = tf.where(mask, t, tf.ones_like(t) * MAX_POSSIBLE_VALUE)
        reduced = tf.math.reduce_max(max_mask, axis=axis, keepdims=keepdims)
        reduced_mask = tf.reduce_any(mask, axis=axis, keepdims=keepdims)
        return tf.where(reduced_mask, reduced, tf.ones_like(t) * default)


def masked_softmax(logits: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    logits_max = masked_reduce_max(logits, mask, 0., axis=-1, keepdims=1)
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


def partial_matrix_reduce_sum(values: tf.Tensor, count: int) -> tf.Tensor:
    """Reduce sum the last dimension in count elements at a time."""
    if len(values.shape) != 2:
        raise ValueError(
            'partial_matrix_reduce_sum only takes tensors for rank=2, got rank {}'.format(len(values.shape)))
    if count == 1:
        return values
    values = _reshape_for_partial_reduction(values, count)
    return tf.reduce_sum(values, axis=-1)


def partial_matrix_reduce_any(mask: tf.Tensor, count: int) -> tf.Tensor:
    """Reduce the last dimension of a mask in count elements at a time."""
    if len(mask.shape) != 2:
        raise ValueError(
            'partial_matrix_reduce_sum only takes tensors for rank=2, got rank {}'.format(len(mask.shape)))
    if count == 1:
        return mask
    mask = _reshape_for_partial_reduction(mask, count)
    return tf.reduce_any(mask, axis=-1)


def _reshape_for_partial_reduction(values: tf.Tensor, count: int) -> tf.Tensor:
    if len(values.shape) != 2:
        raise ValueError(
            '_reshape_for_partial_reduction only takes tensors for rank=2, got rank {}'.format(len(values.shape)))
    mod_count = values.shape[1] % count
    num_blocks = values.shape[1] // count
    if mod_count != 0:
        values = values[:, 0:num_blocks * count]
    return tf.reshape(values, [values.shape[0], num_blocks, count])


def masked_reduce_mean(t: tf.Tensor, mask: tf.Tensor, default: float,
                       axis=None, keepdims=None, name=None) -> tf.Tensor:
    with tf.name_scope(name=name or 'masked_reduce_mean'):
        ones = tf.ones_like(t)
        zeros = tf.zeros_like(t)
        masked_t = tf.where(mask, t, zeros)
        reduced_sum = tf.math.reduce_sum(
            masked_t, axis=axis, keepdims=keepdims)
        reduced_count = tf.math.reduce_sum(
            tf.where(mask, ones, zeros), axis=axis, keepdims=keepdims)
        reduced_count = tf.maximum(reduced_count, tf.ones_like(reduced_count))
        mean = reduced_sum / reduced_count
        mask_any = tf.reduce_any(mask, axis=axis, keepdims=keepdims)
        mean = tf.where(mask_any, mean, default * tf.ones_like(mean))
        return mean
