# optimizations contains many optimization functions, written for Tensorflow
import math
import tensorflow as tf
from typing import Dict, List, Text, Tuple

import tf_lib


def adjust_pct_gain(values: tf.Tensor, loss_cost: float) -> tf.Tensor:
    """Adjusts negative pct gains by multiplying by loss_cost."""
    ones = tf.ones_like(values)
    return values * tf.where(values < 0.0, loss_cost * ones, ones)


def adjust_log_gain(values: tf.Tensor, loss_cost: float) -> tf.Tensor:
    """Adjusts negative log gains by summing by log(loss_cost)."""
    ones = tf.ones_like(values)
    zeros = tf.zeros_like(values)

    # Calculate log gain assuming all are losses.
    log_adjusted_gain = tf.math.log(
        loss_cost * tf.math.exp(values) + (1.0 - loss_cost))
    return tf.where(values < 0.0, log_adjusted_gain, values)


def total_gain_from_log_gains(values: tf.Tensor) -> tf.Tensor:
    """Sum log gains and take the exponetial."""
    sum = tf.reduce_sum(values, axis=-1)
    return tf.math.exp(sum)


def value_of_argmax_prev_value(argmax_values: tf.Tensor, values: List[tf.Tensor],
                               mask: tf.Tensor, transposed: bool, default: float) -> List[tf.Tensor]:
    """Returns the value of the column that is the max of the previous row, and adjusted mask.

    For instance:

    [[ 1 ,  4 , 2]    # -> Max 4, so select the 3 on next row. For this row uses default=0.0
     [ 7 , _3_, 1]    # -> Max 7, so select the 8 on the next row.
     [_8_,  1,  4]]

    Will return: [3, 8] (it returns nothing for the first value).

    Args:
        argmax_values: matrix (2D tensor) with values on which selection decision is made.
        values: List of tensors from which values are taken.
        mask: matix, same shape as values. True for valid values.
        transposed: if values and mask are transposed. If true, they are transposed before
          being used.
        default: value to use when the selected value is not masked.
    """
    if len(argmax_values.shape) != 2:
        raise ValueError(
            'value_of_argmax_prev_value only takes tensors for rank=2, got rank {}'.format(len(argmax_values.shape)))

    # Transpose per request.
    if transposed:
        argmax_values = tf.transpose(argmax_values)
        values = [tf.transpose(value) for value in values]
        mask = tf.transpose(mask)

    # Replace masked-out values by impossible minimun, so never selected.
    masked_argmax_values = tf.where(
        mask, argmax_values, tf_lib.MIN_POSSIBLE_VALUE * tf.ones_like(argmax_values))

    # Find argmax, discard last row, since it's not used. (each argmax is used to
    # select the following row)
    argmax = tf.expand_dims(tf.math.argmax(masked_argmax_values, axis=-1), 1)
    argmax = argmax[:-1]  # Discard last value, since it's not used.
    batch_dims = len(argmax.shape) - 1

    # Select new_values, skipping the first row (since there is no heuristic for
    # that one)
    new_values = [
        tf.gather_nd(value[1:, :], argmax, batch_dims=batch_dims)
        for value in values]

    # Set default value for when selection no longer exists.
    new_mask = tf.gather_nd(mask[1:, :], argmax, batch_dims=batch_dims)
    default_values = default * tf.ones_like(new_values[0])
    new_values = [tf.where(new_mask, value, default_values)
                  for value in new_values]
    return new_values


# Calculate gains
def greedy_selection_for_period(argmax: tf.Tensor, values: List[tf.Tensor], mask: tf.Tensor, period: int) -> List[tf.Tensor]:
    """Aggregate for period and make selection based on argmax of the previous period."""
    mask = tf_lib.partial_matrix_reduce_any(mask, period)
    argmax = tf_lib.partial_matrix_reduce_sum(argmax, period)
    values = [tf_lib.partial_matrix_reduce_sum(value, period)
              for value in values]
    return value_of_argmax_prev_value(argmax, values, mask, transposed=True, default=0.0)
