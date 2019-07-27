# optimizations contains many optimization functions, written for Tensorflow
from absl import logging
import math
import numpy as np
import tensorflow as tf
from typing import Dict, List, Text, Tuple

import config
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


def total_annualized_gain_from_log_gains(values: tf.Tensor) -> tf.Tensor:
    """Returns annualized (p.a) gains, converted in %."""
    years = values.shape[-1] / float(config.YEARLY_PERIOD_IN_SERIAL)
    sum = tf.reduce_sum(values, axis=-1)
    return 100.0 * tf.math.exp(sum / years) - 100.0


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


def mix(logits, gains, mask, loss_cost):
    """Calculate the gains from the mix of assets given by the logits."""
    assets_mix_logits_tiled = tf.tile(
        tf.expand_dims(logits, axis=0), [gains.shape[0], 1])
    # print(f'    mask.shape={mask.shape}')
    # print(f'    assets_mix_logits_tiled.shape={assets_mix_logits_tiled.shape}')
    assets_mix = tf_lib.masked_softmax(assets_mix_logits_tiled, mask)
    # print(f'    assets_mix=({assets_mix.shape})\n{assets_mix}\n')
    mixed_gains = tf.math.log(tf.reduce_sum(
        assets_mix * tf.math.exp(gains), axis=-1))
    # print(f'    mixed_gains=({mixed_gains.shape})\n{mixed_gains}\n')
    adjusted_mixed_gains = adjust_log_gain(mixed_gains, loss_cost)
    # print(f'    adjusted_mixed_gains=({adjusted_mixed_gains.shape})\n{adjusted_mixed_gains}\n')
    total = total_gain_from_log_gains(mixed_gains)
    total_adjusted = total_gain_from_log_gains(adjusted_mixed_gains)
    return mixed_gains, adjusted_mixed_gains, total, total_adjusted, assets_mix


def optimize_mix(symbols: List[Text], gains: tf.Tensor, mask: tf.Tensor,
                 hparams: Dict[str, float]) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, np.array, np.array]:
    """Find mix that maximizes gains."""
    gains_t = tf.transpose(gains)  # shape=[serials, symbols]
    mask_t = tf.transpose(mask)
    assets_mix_logits = tf.Variable(
        tf.zeros(dtype=tf.float32, shape=[len(symbols)]),
        trainable=True)

    # Maximize adjusted_mixed_gains ...
    loss_cost = hparams['loss_cost']
    learning_rate = hparams['learning_rate']
    steps = int(hparams['steps'])
    l1_l2_reg = tf.keras.regularizers.L1L2(l1=hparams['l1'], l2=hparams['l2'])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    for train_ii in range(steps):
        with tf.GradientTape() as tape:
            _, _, _, total_adjusted, _ = mix(
                assets_mix_logits, gains_t, mask_t, loss_cost)
            assets_mix = tf.nn.softmax(assets_mix_logits)
            loss = l1_l2_reg(assets_mix) - total_adjusted
            trainable_vars = [assets_mix_logits]
            grads = tape.gradient(loss, trainable_vars)
            # print(f'    dtotal_dlogits={dtotal_dlogits}')
            optimizer.apply_gradients(zip(grads, trainable_vars))
            # assets_mix_logits.assign_sub(dtotal_dlogits * learning_rate)
            if train_ii % 100 == 0:
                logging.info(f'Training: step={train_ii}, total_adjusted={total_adjusted:.4f}')

    mixed_gains, adjusted_mixed_gains, total, total_adjusted, assets_mix = mix(
        assets_mix_logits, gains_t, mask_t, loss_cost)
    return (mixed_gains, adjusted_mixed_gains, total, total_adjusted, assets_mix_logits.numpy(), assets_mix.numpy())

    # print(f'*** total={total:.4f}, total_adjusted={total_adjusted:.4f},\n' +
    #       f'    assets_mix: logits={assets_mix_logits.numpy()} last={assets_mix[-1]}\n')
    # for ii in range(10):
    #     rows = config.YEARLY_PERIOD_IN_SERIAL * (ii + 1)
    #     last_year_gains = mixed_gains[-rows:]
    #     last_year_gains = total_gain_from_log_gains(last_year_gains)
    #     last_year_adjusted_gains = adjusted_mixed_gains[-rows:]
    #     last_year_adjusted_gains = total_gain_from_log_gains(
    #         last_year_adjusted_gains)
    #     last_year_assets_mix = assets_mix[-rows].numpy()
    #     np.set_printoptions(precision=4, linewidth=120, threshold=100)
    #     print(f'    last {ii+1} year(s): total={last_year_gains:.4f}, total_adjusted={last_year_adjusted_gains:.4f}\n' +
    #           f'                         mix={last_year_assets_mix}')

    #     np.set_printoptions(precision=4, linewidth=120, threshold=100)
    #     print(f'    last {ii+1} year(s): total={last_year_gains:.4f}, total_adjusted={last_year_adjusted_gains:.4f}\n' +
    #           f'                         mix={last_year_assets_mix}')
