#!/usr/bin/env python

# etf_ts.py will download and generate some basic statistics about given list o
# assets (the idea is to focus on ETFs).
#
# It downloads the info from WorldTradingData.com (it seems a great service!),
# and caches it locally to disk. If data is > 10 days old, it will reload.
#
# It requires one to create an account in WorldTradingData.com and pasting
# your key to .../data/WTD_API_KEY.txt.
#
# Example of how to run it:
#
#     $ ./etf_ts.py --data $(pwd)/data
#
# Eventually it will train a model to optimze a balanced investment ... when
# free time allows.
#
# import tensorflow as tf
# from tensorflow import keras

from absl import app
from absl import flags
from absl import logging
import sys
import numpy as np
import tensorflow as tf
from typing import Dict, List, Text

import asset_measures
import config
import data_manager
import dense_measures
import optimizations
import tf_lib

# Flags
FLAGS = flags.FLAGS
flags.DEFINE_string(
    'data', None,
    'Base path where to store historical marked data. Can be shared across models')
flags.DEFINE_integer('max_age_days', 10,
                     'Maximum number of days before updating cached data.')
flags.DEFINE_list(
    'symbols', None,
    'List of sBase path where to store historical marked data. Can be shared across models')
flags.DEFINE_bool(
    'force_recalc', False,
    'If true forces recalculation of derived values.')
flags.DEFINE_float(
    'loss_cost', 1.1,
    'Relative cost of losses compared to gain: the larger this number the larger the penalty for volatility.')
flags.DEFINE_list(
    'stats', 'per_asset,greedy,average,optimal_mix,mix,selection',
    'List of stats to output. A selection of: per_asset, etc.'
)
flags.DEFINE_integer('mix_steps', 300,
                     'Number of steps to optimize each period in mixed strategy.')
flags.mark_flag_as_required('data')

flags.DEFINE_float(
    'mix_training_period', 1.0,
    'For the mix strategy, the amount of years (fractional) used for training before selecting.')
flags.DEFINE_float(
    'mix_applying_period', 0.25,
    'For the mix strategy, the amount of years (fractional) the strategy is applied after optimized. '
    'Defines how often one will have to re-invest.')


def main(argv):
    del argv  # Unused.

    # Select and sort symbols.
    symbols = FLAGS.symbols
    if symbols is None:
        symbols = config.TICKERS
    symbols = sorted(symbols)

    # Download data or reload it from disk cache.
    dmgr = data_manager.DataManager(FLAGS.data)
    for symbol in symbols:
        dmgr.DownloadRawData(symbol, max_age_days=FLAGS.max_age_days)

    # Calculate independent (from each other) derived information if not loaded from cache.
    for symbol in symbols:
        if not asset_measures.HasDerivedValues(dmgr.data[symbol]) or FLAGS.force_recalc:
            logging.info(f'Calculating derived values for {symbol}')
            dmgr.data[symbol] = asset_measures.AddDerivedValues(
                dmgr.data[symbol], symbol)
            dmgr.SaveData(symbol)

    # Calculate dense ordered arrays.
    fields, mask, all_serials = dense_measures.DenseMeasureMatrices(
        dmgr, symbols)

    # Extra metrics calculated in Tensorflow
    fields['AdjustedLogDailyGain'] = optimizations.adjusted_log_gains(
        fields['LogDailyGain'], FLAGS.loss_cost)

    # Print out gains for each symbol.
    # Header of all outputs.
    print('Symbol,Gain,Adjusted Gain,Initial,Final,Description')
    if 'average' in FLAGS.stats:
        average(symbols, mask, fields)
    if 'greedy' in FLAGS.stats:
        greedy(symbols, mask, fields)
    if 'optimal_mix' in FLAGS.stats:
        optimal_mix(symbols, mask, fields)
    if 'mix' in FLAGS.stats:
        mix_previous_period(symbols, mask, fields, all_serials)
    if 'per_asset' in FLAGS.stats:
        per_asset_gains(symbols, mask, fields)
    if 'selection' in FLAGS.stats:
        assets_selection(symbols, mask, fields, all_serials)


def per_asset_gains(symbols: List[Text], mask: tf.Tensor, fields: Dict[Text, tf.Tensor]) -> None:
    """Print basic assets data (CSV) and per assets gains and adjusted gains."""
    # Find opening and closing values of each asset.
    open_t = tf.transpose(fields['Open'])
    close_t = tf.transpose(fields['Open'])
    mask_t = tf.transpose(mask)
    initial_values = tf_lib.masked_first(open_t, mask_t)
    final_values = tf_lib.masked_last(close_t, mask_t)

    # Total (adjusted) gain for each asset.
    log_gains = fields['LogDailyGain']
    adjusted_log_gains = fields['AdjustedLogDailyGain']
    log_gains = log_gains[-config.REPORT_PERIOD:, :]
    adjusted_log_gains = adjusted_log_gains[-config.REPORT_PERIOD:, :]
    total_gain = optimizations.total_annualized_gain_from_log_gains(log_gains)
    total_adjusted_gain = optimizations.total_annualized_gain_from_log_gains(
        adjusted_log_gains)

    # Print it out.
    for symbol_idx, symbol in enumerate(symbols):
        gain = total_gain[symbol_idx]
        adjusted_gain = total_adjusted_gain[symbol_idx]

        initial_value = initial_values[symbol_idx]
        final_value = final_values[symbol_idx]
        print(f'{symbol},{gain:.4f},{adjusted_gain:.4f},{initial_value:.2f},{final_value:.2f}')


def average(symbols: List[Text], mask: tf.Tensor, fields: Dict[Text, tf.Tensor]) -> None:
    log_gains = fields['LogDailyGain']
    log_gains = tf.where(mask, log_gains, tf.zeros_like(
        log_gains, dtype=tf.float32))

    partial_gains: List[float] = []
    partial_adjusted_gains: List[float] = []
    logits = tf.zeros(dtype=tf.float32, shape=tf.shape(log_gains)[-1])

    for ii in range(config.REPORT_PERIOD_YEARS):
        start_idx = -(ii + 1) * config.YEARLY_PERIOD_IN_SERIAL
        end_idx = -ii * config.YEARLY_PERIOD_IN_SERIAL
        if ii == 0:
            mask_year = mask[start_idx:, :]
            log_gains_year = log_gains[start_idx:, :]
        else:
            mask_year = mask[start_idx:end_idx, :]
            log_gains_year = log_gains[start_idx:end_idx, :]
        # print(f'mask_year.shape={mask_year.shape}, log_gains_year.shape={log_gains_year.shape}\n{log_gains_year}')

        # Find mask for logits of symbos used.
        symbols_used = tf.reduce_any(mask_year, axis=0)

        # Find gains for the year.
        mix_gain, adjusted_mix_gain = optimizations.mix_gain(
            logits, symbols_used, log_gains_year, FLAGS.loss_cost)
        partial_gains.append(mix_gain.numpy())
        partial_adjusted_gains.append(adjusted_mix_gain.numpy())

    # print(f'partial_gains={partial_gains}')
    # print(f'partial_adjusted_gains={partial_adjusted_gains}')
    average = np.mean(np.array(partial_gains))
    average = 100.0 * (average - 1.0)
    adjusted_average = np.mean(np.array(partial_adjusted_gains))
    adjusted_average = 100.0 * (adjusted_average - 1.0)
    print(f'_Average (last {config.REPORT_PERIOD_YEARS} years),{average:.4f},{adjusted_average:.4f},,,Gains (p.a.), readjusted every year.')


def optimal_mix(symbols: List[Text], mask: tf.Tensor, fields: Dict[Text, tf.Tensor]) -> None:
    log_gains = fields['LogDailyGain']
    log_gains = tf.where(mask, log_gains, tf.zeros_like(log_gains))
    hparams = {
        'steps': 1000,
        'learning_rate': 1.0e-1,
        'loss_cost': FLAGS.loss_cost,
        'l1': 1e-2,
        'l2': 1e-5,
    }
    (mix_gain, adjusted_mix_gain, mix_logits, mix) = (
        optimizations.optimize_mix(symbols, symbols_mask, log_gains, hparams))

    # Take only last 10 years:
    last_year_gains = mixed_gains[-config.REPORT_PERIOD:]
    total = optimizations.total_gain_from_log_gains(last_year_gains)
    last_year_adjusted_gains = adjusted_mixed_gains[-config.REPORT_PERIOD:]
    total_adjusted = optimizations.total_gain_from_log_gains(
        last_year_adjusted_gains)
    print(f'_Oracle Mix (last {config.REPORT_PERIOD_YEARS} years),{total:.4f},{total_adjusted:.4f},,,' +
          f'Best fixed mixed logist (softmax-ed according to availability of asset).')


def mix_previous_period(symbols: List[Text], mask: tf.Tensor, fields: Dict[Text, tf.Tensor], all_serials: List[int]) -> None:
    log_gains = tf.convert_to_tensor(fields['LogDailyGain'], dtype=tf.float32)
    hparams = {
        'steps': FLAGS.mix_steps,
        'learning_rate': 0.1,  # 3.3e-2,
        'loss_cost': FLAGS.loss_cost,
        'l1': 1e-3,
        'l2': 1e-6,
    }
    all_gains: List[tf.Tensor] = []
    all_adjusted_gains: List[tf.Tensor] = []
    apply_cycles = int(config.YEARLY_PERIOD_IN_SERIAL *
                       FLAGS.mix_applying_period)

    for last_ii_year in range(config.REPORT_PERIOD // apply_cycles):
        # Identify range where to apply new mix.
        apply_start = (-last_ii_year - 1) * apply_cycles
        apply_start_date = asset_measures.SerialDateToString(
            all_serials[apply_start])
        apply_end = -last_ii_year * apply_cycles
        apply_end_date = asset_measures.SerialDateToString(
            all_serials[min(apply_end, -1)])
        logging.info(f'Calculating mix for range {apply_start_date} ({apply_start}) to {apply_end_date} ({apply_end})')
        if apply_end == 0:
            apply_gains = log_gains[apply_start:, :]
            apply_mask = mask[apply_start:, :]
        else:
            apply_gains = log_gains[apply_start:apply_end, :]
            apply_mask = mask[apply_start:apply_end, :]
        # print(f'gains.shape={log_gains.shape}, apply_gains.shape={apply_gains.shape}')

        # Find best mix based on last (config.YEARLY_PERIOD_IN_SERIAL * FLAGS.mix_training_period) cycles (a few years).
        train_end = apply_start
        train_start = train_end - \
            int(config.YEARLY_PERIOD_IN_SERIAL * FLAGS.mix_training_period)
        train_gains = log_gains[train_start:train_end, :]
        # print(f'gains.shape={log_gains.shape}, train_gains.shape={train_gains.shape}')
        train_mask = mask[train_start:train_end, :]
        # print(f'mask.shape={mask.shape}, train_mask.shape={train_mask.shape}')

        # Train and find mix.
        symbols_used = tf.reduce_any(train_mask, axis=0)
        # print(f'symbols_used={symbols_used.numpy()}')

        (mix_gain, adjusted_mix_gain, mix_logits, mix) = (
            optimizations.optimize_mix(symbols, symbols_used, train_gains, hparams))

        # Apply mix to apply gains.
        mix_gain, adjusted_mix_gain = optimizations.mix_gain(
            mix_logits, symbols_used, apply_gains, FLAGS.loss_cost)
        mix_gain = optimizations.annualized_gain(mix_gain, apply_cycles)
        adjusted_mix_gain = optimizations.annualized_gain(
            adjusted_mix_gain, apply_cycles)

        print(f'  Applying it to next period: mix_gain={mix_gain:.4f}, adjusted_mix_gain={adjusted_mix_gain:.4f}')
        all_gains = [mix_gain] + all_gains
        all_adjusted_gains = [adjusted_mix_gain] + all_adjusted_gains

    # all_gains = tf.concat(all_gains, axis=-1)
    # all_adjusted_gains = tf.concat(all_adjusted_gains, axis=-1)
    # # print(f'all_gains={all_gains.shape} {all_gains}')
    # # print(f'all_adjusted_gains={all_adjusted_gains.shape} {all_adjusted_gains}')
    # total = optimizations.total_annualized_gain_from_log_gains(all_gains)
    # total_adjusted = optimizations.total_annualized_gain_from_log_gains(
    #     all_adjusted_gains)
    # print(f'_Mix (last {config.REPORT_PERIOD_YEARS} years),{total:.4f},{total_adjusted:.4f},,,Trained with previous 4 years - adjusted quarterly.')


def assets_selection(symbols: List[Text], mask: tf.Tensor, fields: Dict[Text, tf.Tensor], all_serials: List[int]) -> None:
    log_gains = fields['LogDailyGain']
    hparams = {
        'steps': FLAGS.mix_steps,
        'learning_rate': 3.3e-2,
        'loss_cost': FLAGS.loss_cost,
        'l1': 1e-3,
        'l2': 1e-6,
    }

    # Find best mix based on last (config.YEARLY_PERIOD_IN_SERIAL * FLAGS.mix_training_period) cycles (a few years).
    train_gains = log_gains[:, -
                            int(config.YEARLY_PERIOD_IN_SERIAL * FLAGS.mix_training_period):]
    train_mask = mask[:, -int(config.YEARLY_PERIOD_IN_SERIAL *
                              FLAGS.mix_training_period):]
    (_, _, _, _, assets_logits, assets_mix) = optimizations.optimize_mix(
        symbols, train_gains, train_mask, hparams)
    # mix = assets_mix[-1]
    mix = tf.nn.softmax(assets_logits)
    mix = tf.where(mix > 0.01, mix, tf.zeros_like(mix))
    norm = tf.reduce_sum(mix)
    norm = tf.where(norm == 0, 1.0, norm)
    mix = mix / norm
    pairs = [(-mix[ii], symbol, assets_logits[ii])
             for (ii, symbol) in enumerate(symbols)]
    pairs = sorted(pairs)
    for neg_ratio, symbol, logit in pairs:
        ratio = -neg_ratio
        if ratio > 0.0:
            print(f'{symbol},{100.0*ratio:3.0f}%,{logit:.4g}')


def greedy(symbols: List[Text], mask: tf.Tensor, fields: Dict[Text, tf.Tensor]) -> None:
    """Print total gain for best possible selection (oracle) per day."""
    log_gains = fields['LogDailyGain']
    adjusted_log_gains = fields['AdjustedLogDailyGain']

    period_desc = ['Daily', 'Monthly', 'Yearly']
    argmax_desc = ['Gain', 'Adjusted Gain']
    for period_idx, period in enumerate([1, config.MONTHLY_PERIOD_IN_SERIAL, config.YEARLY_PERIOD_IN_SERIAL]):
        for argmax_idx, argmax in enumerate([log_gains, adjusted_log_gains]):

            chosen_gains, chosen_adjusted_gains = optimizations.greedy_selection_for_period(
                argmax, [log_gains, adjusted_log_gains], mask, period)

            chosen_gains = chosen_gains[-config.REPORT_PERIOD:]
            chosen_adjusted_gains = chosen_adjusted_gains[-config.REPORT_PERIOD:]

            total = optimizations.total_annualized_gain_from_log_gains(
                chosen_gains)
            total_adjusted = optimizations.total_annualized_gain_from_log_gains(
                chosen_adjusted_gains)
            print(f'_Greedy {period_desc[period_idx]} {argmax_desc[argmax_idx]},{total:.4f},{total_adjusted:.4f},,,' +
                  f'Selecting asset with best {argmax_desc[argmax_idx]} yield of last {period_desc[period_idx]} period. ' +
                  f' Gains (p.a.) of last {config.REPORT_PERIOD_YEARS}')


if __name__ == '__main__':
    app.run(main)
