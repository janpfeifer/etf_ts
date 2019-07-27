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

    # Convert dense arrays to transposed tensors.
    mask = tf.transpose(tf.constant(mask, dtype=tf.bool))
    for field in fields.keys():
        fields[field] = tf.transpose(
            tf.constant(fields[field], dtype=tf.float32))

    # Extra metrics calculated in Tensorflow
    fields['LogAdjustedGains'] = optimizations.adjust_log_gain(
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
    open_ = fields['Open']
    close = fields['Open']
    initial_values = tf_lib.masked_first(open_, mask)
    final_values = tf_lib.masked_last(close, mask)

    log_gains = fields['LogDailyGain']
    log_adjusted_gains = fields['LogAdjustedGains']

    log_gains = log_gains[:, -config.REPORT_PERIOD:]
    log_adjusted_gains = log_adjusted_gains[:, -config.REPORT_PERIOD:]

    total_gains = optimizations.total_annualized_gain_from_log_gains(log_gains)
    total_adjusted_gains = optimizations.total_annualized_gain_from_log_gains(
        log_adjusted_gains)

    for symbol_idx, symbol in enumerate(symbols):
        gain = total_gains[symbol_idx]
        adjusted_gain = total_adjusted_gains[symbol_idx]

        initial_value = initial_values[symbol_idx]
        final_value = final_values[symbol_idx]
        print(f'{symbol},{gain:.4f},{adjusted_gain:.4f},{initial_value:.2f},{final_value:.2f}')


def greedy(symbols: List[Text], mask: tf.Tensor, fields: Dict[Text, tf.Tensor]) -> None:
    """Print total gain for best possible selection (oracle) per day."""
    log_gains = fields['LogDailyGain']
    log_adjusted_gains = fields['LogAdjustedGains']

    period_desc = ['Daily', 'Monthly', 'Yearly']
    argmax_desc = ['Gain', 'Adjusted Gain']
    for period_idx, period in enumerate([1, config.MONTHLY_PERIOD_IN_SERIAL, config.YEARLY_PERIOD_IN_SERIAL]):
        for argmax_idx, argmax in enumerate([log_gains, log_adjusted_gains]):

            chosen_gains, chosen_adjusted_gains = optimizations.greedy_selection_for_period(
                argmax, [log_gains, log_adjusted_gains], mask, period)

            chosen_gains = chosen_gains[-config.REPORT_PERIOD:]
            chosen_adjusted_gains = chosen_adjusted_gains[-config.REPORT_PERIOD:]

            total = optimizations.total_annualized_gain_from_log_gains(
                chosen_gains)
            total_adjusted = optimizations.total_annualized_gain_from_log_gains(
                chosen_adjusted_gains)
            print(f'_Greedy {period_desc[period_idx]} {argmax_desc[argmax_idx]},{total:.4f},{total_adjusted:.4f},,,' +
                  f'Selecting asset with best {argmax_desc[argmax_idx]} yield of last {period_desc[period_idx]} period. ' +
                  f' Gains (p.a.) of last {config.REPORT_PERIOD_YEARS}')


def average(symbols: List[Text], mask: tf.Tensor, fields: Dict[Text, tf.Tensor]) -> None:
    log_gains = fields['LogDailyGain']
    gains = tf.math.exp(log_gains)

    chosen_gains = tf.math.log(tf.transpose(tf_lib.masked_reduce_mean(
        tf.transpose(gains), tf.transpose(mask), 1.0, axis=-1)))
    chosen_adjusted_gains = optimizations.adjust_log_gain(
        chosen_gains, FLAGS.loss_cost)

    # Select last 10 years.
    chosen_gains = chosen_gains[-config.REPORT_PERIOD:]
    chosen_adjusted_gains = chosen_adjusted_gains[-config.REPORT_PERIOD:]

    total = optimizations.total_annualized_gain_from_log_gains(chosen_gains)
    total_adjusted = optimizations.total_annualized_gain_from_log_gains(
        chosen_adjusted_gains)
    print(f'_Value of an averaged account (last {config.REPORT_PERIOD_YEARS} years),{total:.4f},{total_adjusted:.4f},,,Gains (p.a.) of an even mix of all assets tracked.')


def optimal_mix(symbols: List[Text], mask: tf.Tensor, fields: Dict[Text, tf.Tensor]) -> None:
    log_gains = fields['LogDailyGain']
    hparams = {
        'steps': 1000,
        'learning_rate': 1.0e-1,
        'loss_cost': FLAGS.loss_cost,
        'l1': 1e-2,
        'l2': 1e-5,
    }
    (mixed_gains, adjusted_mixed_gains, total, total_adjusted, assets_mix_logits, assets_mix) = (
        optimizations.optimize_mix(symbols, log_gains, mask, hparams))

    # Take only last 10 years:
    last_year_gains = mixed_gains[-config.REPORT_PERIOD:]
    total = optimizations.total_gain_from_log_gains(last_year_gains)
    last_year_adjusted_gains = adjusted_mixed_gains[-config.REPORT_PERIOD:]
    total_adjusted = optimizations.total_gain_from_log_gains(
        last_year_adjusted_gains)
    print(f'_Oracle Mix (last {config.REPORT_PERIOD_YEARS} years),{total:.4f},{total_adjusted:.4f},,,' +
          f'Best fixed mixed logist (softmax-ed according to availability of asset).')


def mix_previous_period(symbols: List[Text], mask: tf.Tensor, fields: Dict[Text, tf.Tensor], all_serials: List[int]) -> None:
    log_gains = fields['LogDailyGain']
    hparams = {
        'steps': FLAGS.mix_steps,
        'learning_rate': 3.3e-2,
        'loss_cost': FLAGS.loss_cost,
        'l1': 1e-3,
        'l2': 1e-6,
    }
    all_gains: List[tf.Tensor] = []
    all_adjusted_gains: List[tf.Tensor] = []
    for last_ii_year in range(config.REPORT_PERIOD // config.APPLYING_PERIOD):
        # Identify range where to apply new mix.
        apply_start = (-last_ii_year - 1) * config.APPLYING_PERIOD
        apply_start_date = asset_measures.SerialDateToString(
            all_serials[apply_start])
        apply_end = -last_ii_year * config.APPLYING_PERIOD
        apply_end_date = asset_measures.SerialDateToString(
            all_serials[min(apply_end, -1)])
        logging.info(f'Calculating mix for range {apply_start_date} ({apply_start}) to {apply_end_date} ({apply_end})')
        if apply_end == 0:
            apply_gains = log_gains[:, apply_start:]
            apply_mask = mask[:, apply_start:]
        else:
            apply_gains = log_gains[:, apply_start:apply_end]
            apply_mask = mask[:, apply_start:apply_end]
        # print(f'gains.shape={log_gains.shape}, apply_gains.shape={apply_gains.shape}')

        # Find best mix based on last config.TRAINING_PERIOD cycles (a few years).
        train_end = apply_start
        train_start = train_end - config.TRAINING_PERIOD
        train_gains = log_gains[:, train_start:train_end]
        # print(f'gains.shape={log_gains.shape}, train_gains.shape={train_gains.shape}')
        train_mask = mask[:, train_start:train_end]
        # print(f'mask.shape={mask.shape}, train_mask.shape={train_mask.shape}')

        # Train and find mix.
        (_, _, _, _, assets_mix_logits, _) = optimizations.optimize_mix(
            symbols, train_gains, train_mask, hparams)

        # Apply mix to apply gains.
        mixed_gains, adjusted_mixed_gains, _, _, _ = optimizations.mix(
            assets_mix_logits,
            tf.transpose(apply_gains), tf.transpose(apply_mask),
            FLAGS.loss_cost)
        all_gains = [mixed_gains] + all_gains
        all_adjusted_gains = [adjusted_mixed_gains] + all_adjusted_gains

    all_gains = tf.concat(all_gains, axis=-1)
    all_adjusted_gains = tf.concat(all_adjusted_gains, axis=-1)
    # print(f'all_gains={all_gains.shape} {all_gains}')
    # print(f'all_adjusted_gains={all_adjusted_gains.shape} {all_adjusted_gains}')
    total = optimizations.total_annualized_gain_from_log_gains(all_gains)
    total_adjusted = optimizations.total_annualized_gain_from_log_gains(
        all_adjusted_gains)
    print(f'_Mix (last {config.REPORT_PERIOD_YEARS} years),{total:.4f},{total_adjusted:.4f},,,Trained with previous 4 years - adjusted quarterly.')


def assets_selection(symbols: List[Text], mask: tf.Tensor, fields: Dict[Text, tf.Tensor], all_serials: List[int]) -> None:
    log_gains = fields['LogDailyGain']
    hparams = {
        'steps': FLAGS.mix_steps,
        'learning_rate': 3.3e-2,
        'loss_cost': FLAGS.loss_cost,
        'l1': 1e-3,
        'l2': 1e-6,
    }

    # Find best mix based on last config.TRAINING_PERIOD cycles (a few years).
    train_gains = log_gains[:, -config.TRAINING_PERIOD:]
    train_mask = mask[:, -config.TRAINING_PERIOD:]
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


if __name__ == '__main__':
    app.run(main)
