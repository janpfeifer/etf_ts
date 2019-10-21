# optimizations contains many optimization functions, written for Tensorflow
from absl import logging
import math
import numpy as np
import tensorflow as tf
from typing import Dict, List, Text, Tuple

import config
import tf_lib


def mix_gain(logits: tf.Tensor, logits_mask: tf.Tensor, log_gains: tf.Tensor, 
             loss_cost: float, gain_power: float) -> Tuple[tf.Tensor, tf.Tensor]:
    """Calculate the gains from the mix of assets given by the logits.

    Args:
        logits: Logits on how one wants to invest on. Shape is `[num_assets]`.
        logits_mask: Which logits are in consideration. Shape is `[num_assets]`.
        log_gains: Log gains reported for period, shape `[num_cycles, num_assets]`,
          where typically `num_cycles` is the number of days, but they could
          have been aggregated in larger periods. Gains should be set to 0
          (that is, no change) for days/cycles when asset can't be negotiated.
        loss_cost: Penalty assigned for losses.
        gain_power: Power associate to gains -- favouring smoother gains.

    Returns:
        mix_gain: Scalar of total accumulated gain of mix, loss penalty not considered.
            It is in the form of a ration with respect to the capital invested (so 1.5 
            would mean a gain of 50% in the period)
        adjusted_mix_gain: Scalar of total accumulated gain of mix,
            adjusted by `loss_cost`.

    """
    weights = tf_lib.masked_softmax(logits, logits_mask)
    #print(f'weights={weights}')
    log_gains_cum = tf.cumsum(log_gains, axis=0, exclusive=True)
    values = weights * tf.math.exp(log_gains_cum)
    #print(f'values: shape={values.shape}, values={values.numpy()}')
    sum_values = tf.reduce_sum(values, axis=-1)
    #print(f'sum_values: shape={sum_values.shape}, values={sum_values.numpy()}')
    shifted_sum_values = tf.concat([[1.0], sum_values[:-1]], axis=0)
    #print(f'shifted_sum_values: shape={shifted_sum_values.shape}, values={shifted_sum_values.numpy()}')

    # Check cost of losses.
    all_assets_gains_ratio = sum_values / shifted_sum_values
    mix_gains = all_assets_gains_ratio - 1.0
    #print(f'mix_gains: shape={mix_gains.shape}, values={mix_gains.numpy()}')
    loss_cost_on_gains = tf.where(
        mix_gains > 0.0, 0.0, (loss_cost-1) * mix_gains)
    #print(f'loss_cost: shape={loss_cost_on_gains.shape}, values={loss_cost_on_gains.numpy()}')

    # Cost of wins: we want to favor
    if gain_power == 1:
        adjustment_gains = None
    else: 
        #print(f'gains_ratio: shape={all_assets_gains_ratio.shape}, values={all_assets_gains_ratio.numpy()}')
        all_assets_log_gains = tf.math.log(all_assets_gains_ratio)
        #print(f'log_gains: shape={all_assets_log_gains.shape}, values={all_assets_log_gains.numpy()}')
        adjustment_gains = tf.where(
            all_assets_log_gains < 0, 0, tf.math.pow(all_assets_log_gains, gain_power) - all_assets_log_gains)
        #print(f'gain_cost: shape={adjustment_gains.shape}, values={adjustment_gains.numpy()}')
        adjustment_gains = tf.exp(tf.reduce_sum(adjustment_gains))
        #print(f'gain_cost: {adjustment_gains}')

    mix_gain = tf.reduce_sum(values[-1])
    adjustment = tf.exp(tf.reduce_sum(tf.math.log(1.0 + loss_cost_on_gains)))
    if adjustment_gains is not None:
        adjustment *= adjustment_gains
    #print(f'adjustment={adjustment}')
    adjusted_mix_gain = adjustment * mix_gain
    return mix_gain, adjusted_mix_gain


def optimize_mix(symbols: List[Text], logits_mask: tf.Tensor, log_gains: tf.Tensor,
                 hparams: Dict[str, float]) -> Tuple[float, float, np.array, np.array]:
    """Find mix that maximizes gains."""
    assets_mix_logits = tf.Variable(
        tf.zeros(dtype=tf.float32, shape=[len(symbols)]),
        trainable=True)

    # Maximize adjusted_mixed_gains ...
    loss_cost = float(hparams['loss_cost'])
    learning_rate = hparams['learning_rate']
    steps = int(hparams['steps'])
    l1_l2_reg = tf.keras.regularizers.L1L2(l1=hparams['l1'], l2=hparams['l2'])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    for train_ii in range(steps):
        with tf.GradientTape() as tape:
            mix_gain_, adjusted_mix_gain = mix_gain(
                assets_mix_logits, logits_mask, log_gains, loss_cost)
            assets_mix = tf_lib.masked_softmax(assets_mix_logits, logits_mask)
            loss = l1_l2_reg(assets_mix) - adjusted_mix_gain
            trainable_vars = [assets_mix_logits]
            grads = tape.gradient(loss, trainable_vars)
            optimizer.apply_gradients(zip(grads, trainable_vars))
            if train_ii % 100 == 0:
                logging.info(f'Training: step={train_ii}, adjusted_mix_gain={adjusted_mix_gain:.4f}')

    return (mix_gain_.numpy(), adjusted_mix_gain.numpy(), assets_mix_logits.numpy(), assets_mix.numpy())


def annualized_gain(gain: float, cycles: int) -> float:
    """Convert gain to annualized gain.

    Args:
        gain: Multiplicative gain, so 1.5 for 50% gain, accumulated
            over `cycles` trading days.
        cycles: Number of cycles over which gain was calculated.

    Returns:
        Annualized (p.a.) gain, as a multiplicative facor.
    """
    ratio = float(config.YEARLY_PERIOD_IN_SERIAL) / float(cycles)
    return math.exp(math.log(gain) * ratio)


def adjusted_pct_gains(pct_gains: tf.Tensor, loss_cost: float) -> tf.Tensor:
    """Adjusts negative pct gains by multiplying by loss_cost.

    Args:
        pct_gains: Gains in form of percentage points reported for period,
            shape `[num_cycles, num_assets]`, where typically `num_cycles`
            is the number of days, but they could have been aggregated in
            larger periods. Gains should be set to 0 (that is, no change)
            for days/cycles when asset can't be negotiated.
        loss_cost: Penalty assigned for losses.
    Returns:
        adjusted_pct_gains: Tensor with same shape as pct_gains
            `[num_cycles, num_assets]`, but with its negative values adjusted
            for the loss_cost penalty.
    """
    ones = tf.ones_like(pct_gains)
    return pct_gains * tf.where(pct_gains < 0.0, loss_cost * ones, ones)


def adjusted_log_gains(log_gains: tf.Tensor, loss_cost: float, gain_power: float) -> tf.Tensor:
    """Adjusts negative log gains by summing by log(loss_cost).

    Args:
        log_gains: Log gains reported for period, shape `[num_cycles, num_assets]`,
            where typically `num_cycles` is the number of days, but they could
            have been aggregated in larger periods. Gains should be set to 0
            (that is, no change) for days/cycles when asset can't be negotiated.
        loss_cost: Penalty assigned for losses (a multiplier on the pct_gains,
            applied on negative values only).
    Returns:
        adjusted_log_gains: Tensor with same shape as pct_gains
            `[num_cycles, num_assets]`, but with its negative values adjusted
            for the loss_cost penalty.
    """
    # Calculate adjusted log gain assuming all are losses.
    adjusted_neg = tf.math.log(loss_cost * (tf.math.exp(log_gains) - 1.0) + 1.0)
    if gain_power != 1:
        adjusted_pos = tf.math.power(tf.math.abs(log_gains), gain_power)
    else:
        adjusted_pos = log_gains

    # Use adjusted values only when log_gain < 0.
    return tf.where(log_gains < 0.0, adjusted_neg, adjusted_pos)


def total_gain_from_log_gains(log_gains: tf.Tensor) -> tf.Tensor:
    """Sum log gains and take the exponetial."""
    sum = tf.reduce_sum(log_gains, axis=0)
    return tf.math.exp(sum)


def total_annualized_gain_from_log_gains(log_gains: tf.Tensor) -> tf.Tensor:
    """Returns annualized (p.a) gains, converted into %."""
    years = log_gains.shape[0] / float(config.YEARLY_PERIOD_IN_SERIAL)
    sum = tf.reduce_sum(log_gains, axis=0)
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
