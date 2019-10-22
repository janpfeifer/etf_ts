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
from typing import Dict, List, Text, Tuple

import asset_measures
import config
import config_ib
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
flags.DEFINE_integer('symbols_max_count', 0,
                     'Used for testing. If >0 take the first --symbols_max_count symbols only.')
flags.DEFINE_bool(
    'force_recalc', False,
    'If true forces recalculation of derived values.')
flags.DEFINE_float(
    'loss_cost', 1.3,
    'Relative cost of losses compared to gain: the larger this number the larger the penalty for volatility.')
flags.DEFINE_float(
    'tax_on_dividends', asset_measures.TAX_ON_DIVIDENDS_PERCENTAGE,
    'Percentage of dividends (0 to 1.0) loss due to taxes.')
flags.DEFINE_float(
    'gain_power', 1,
    'Log gains when positive are powered by this value. Values < 1.0 will tend to make choices more smooth.')

flags.DEFINE_list(
    'stats', 'per_asset,greedy,average,mix,selection',
    'List of stats to output. A selection of: per_asset, etc.'
)
flags.DEFINE_integer('mix_steps', 300,
                     'Number of steps to optimize each period in mixed strategy.')
flags.mark_flag_as_required('data')


flags.DEFINE_float(
    'mix_training_period', 2.0,
    'For the mix strategy, the amount of years (fractional) used for training before selecting.')
flags.DEFINE_float(
    'mix_applying_period', 0.25,
    'For the mix strategy, the amount of years (fractional) the strategy is applied after optimized. '
    'Defines how often one will have to re-invest.')
flags.DEFINE_integer(
    'max_days', None,
    'If set, look at most at the latest given number of days. Otherwise look at everything.')


def main(argv):
    del argv  # Unused.

    tf_lib.config_gpu()

    asset_measures.TAX_ON_DIVIDENDS_PERCENTAGE = FLAGS.tax_on_dividends
    dense_measures.MAX_DAYS = FLAGS.max_days

    # Select and sort symbols.
    symbols = config_ib.extract_ib_symbols(FLAGS.data, FLAGS.max_age_days)
    if FLAGS.symbols is not None:
        symbols = FLAGS.symbols

    # Download data or reload it from disk cache.
    dmgr = data_manager.DataManager(FLAGS.data)
    symbols = dmgr.DownloadRawDataForList(
        symbols, max_age_days=FLAGS.max_age_days)

    # Calculate independent (from each other) derived information if not loaded from cache.
    num_symbols = len(symbols)
    for ii, symbol in enumerate(symbols):
        if not asset_measures.HasDerivedValues(dmgr.data[symbol]) or FLAGS.force_recalc:
            logging.info(f'Calculating derived values for {symbol} ({num_symbols - ii} missing)')
            dmgr.data[symbol] = asset_measures.AddDerivedValues(
                dmgr.data[symbol], dmgr.dividends[symbol], symbol)
            dmgr.SaveData(symbol)

    # Calculate dense ordered arrays.
    fields, mask, all_serials = dense_measures.DenseMeasureMatrices(
        dmgr.data, symbols)

    # Extra metrics calculated in Tensorflow
    fields['AdjustedLogDailyGain'] = optimizations.adjusted_log_gains(
        fields['LogDailyGain'], FLAGS.loss_cost, FLAGS.gain_power)

    # Print out gains for each symbol.
    # Header of all outputs.
    print('Symbol,Gain,Adjusted Gain,Initial,Final,Description')
    if 'average' in FLAGS.stats:
        average(symbols, mask, fields)
    # if 'greedy' in FLAGS.stats:
    #     greedy(symbols, mask, fields)
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
    if tf.reduce_any(tf.math.is_nan(total_gain)) or tf.reduce_any(tf.math.is_nan(total_adjusted_gain)):
        logging.info(f'nan in log_gains: {tf.where(tf.math.is_nan(log_gains))}')
        logging.info(f'nan in adjusted: {tf.where(tf.math.is_nan(adjusted_log_gains))}')
        logging.info('You probably want to remove those symbols with noisy data. '
                     'Run with stats=per_asset, redirect to a file and grep for nan.')

    # Print it out.
    for symbol_idx, symbol in enumerate(symbols):
        gain = total_gain[symbol_idx]
        description = '?'
        if symbol in config_ib.SYMBOL_TO_INFO:
            description = config_ib.SYMBOL_TO_INFO[symbol]['description']
        adjusted_gain = total_adjusted_gain[symbol_idx]

        initial_value = initial_values[symbol_idx]
        final_value = final_values[symbol_idx]
        print(f'{symbol},{gain:.4f},{adjusted_gain:.4f},{initial_value:.2f},{final_value:.2f},{description}')


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


def mix_previous_period(symbols: List[Text], mask: tf.Tensor, fields: Dict[Text, tf.Tensor], all_serials: List[int]) -> None:
    log_gains = tf.convert_to_tensor(fields['LogDailyGain'], dtype=tf.float32)
    hparams = {
        'steps': FLAGS.mix_steps,
        'learning_rate': 0.1,  # 3.3e-2,
        'loss_cost': FLAGS.loss_cost,
        'gain_power': FLAGS.gain_power,
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
        if apply_start + len(all_serials) <= 0:
            logging.info('Not enough data, stopping mix here.')
            break
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
        if train_start + len(all_serials) <= 0:
            logging.info('Not enough data, stopping mix here.')
            break
        print(f'train_start={train_start}, len(all_serials)={len(all_serials)}')
        train_gains = log_gains[train_start:train_end, :]
        # print(f'gains.shape={log_gains.shape}, train_gains.shape={train_gains.shape}')
        train_mask = mask[train_start:train_end, :]
        train_serials = all_serials[train_start:train_end]
        # print(f'mask.shape={mask.shape}, train_mask.shape={train_mask.shape}')

        # Train and find mix.
        # print(f'train_serials={train_serials}')
        # print(f'train_mask={train_mask}')
        symbols_used = tf.constant(dense_measures.SelectSymbolsFromMask(
            train_serials, train_mask), dtype=tf.bool)
        # print(f'symbols_used={symbols_used.numpy()}')

        (mix_gain, adjusted_mix_gain, mix_logits, mix) = (
            optimizations.optimize_mix(symbols, symbols_used, train_gains, hparams))
        selection = _normalize_selection(symbols, mix_logits, symbols_used)
        selection = [f'{symbol}: {ratio:3.1f}%' for ratio, symbol in selection]
        logging.info('  Selection of assets: {}'.format(', '.join(selection)))

        # Apply mix to apply gains.
        mix_gain, adjusted_mix_gain = optimizations.mix_gain(
            mix_logits, symbols_used, apply_gains, FLAGS.loss_cost, FLAGS.gain_power)
        mix_gain = optimizations.annualized_gain(mix_gain, apply_cycles)
        adjusted_mix_gain = optimizations.annualized_gain(
            adjusted_mix_gain, apply_cycles)

        # print(f'  Applying it to next period: mix_gain={mix_gain:.4f}, adjusted_mix_gain={adjusted_mix_gain:.4f}')
        all_gains = [mix_gain] + all_gains
        all_adjusted_gains = [adjusted_mix_gain] + all_adjusted_gains

    all_gains = np.array(all_gains)
    all_adjusted_gains = np.array(all_adjusted_gains)
    mix_gain = np.mean(all_gains)
    adjusted_mix_gain = np.mean(all_adjusted_gains)
    mix_gain = 100.0 * (mix_gain - 1)
    adjusted_mix_gain = 100.0 * (adjusted_mix_gain - 1)

    print(f'_Mix (last {config.REPORT_PERIOD_YEARS} years),' +
          f'{mix_gain:.4f},{adjusted_mix_gain:.4f},,,' +
          f'Trained with preceding {FLAGS.mix_training_period} year(s) ' +
          f'and then applied for {FLAGS.mix_applying_period} year(s).')


def _normalize_selection(symbols: List[Text], logits: tf.Tensor, logits_mask: tf.Tensor) -> List[Tuple[float, Text]]:
    """Select list of assets and its percentual participation after normalization."""
    mix = tf_lib.masked_softmax(logits, logits_mask)
    mix = tf.where(mix >= 0.01, mix, tf.zeros_like(mix))
    norm = tf.reduce_sum(mix)
    norm = tf.where(norm == 0, 1.0, norm)
    mix = 100.0 * mix / norm
    mix = mix.numpy()
    pairs = [(mix[ii], symbol)
             for (ii, symbol) in enumerate(symbols)
             if mix[ii] > 0.0]
    pairs = sorted(pairs, reverse=True)
    return pairs


def assets_selection(symbols: List[Text], mask: tf.Tensor, fields: Dict[Text, tf.Tensor], all_serials: List[int]) -> None:
    log_gains = tf.convert_to_tensor(fields['LogDailyGain'], dtype=tf.float32)
    hparams = {
        'steps': FLAGS.mix_steps,
        'learning_rate': 0.1,
        'loss_cost': FLAGS.loss_cost,
        'gain_power': FLAGS.gain_power,
        'l1': 1e-3,
        'l2': 1e-6,
    }

    # Find best mix based on last (config.YEARLY_PERIOD_IN_SERIAL * FLAGS.mix_training_period) cycles (a few years).
    train_cycles = int(config.YEARLY_PERIOD_IN_SERIAL *
                       FLAGS.mix_training_period)
    train_gains = log_gains[-train_cycles:, :]
    train_mask = mask[-train_cycles:, :]
    train_serials = all_serials[-train_cycles:]
    symbols_used = tf.constant(dense_measures.SelectSymbolsFromMask(
        train_serials, train_mask), dtype=tf.bool)
    (mix_gain, adjusted_mix_gain, mix_logits, mix) = (
        optimizations.optimize_mix(symbols, symbols_used, train_gains, hparams))
    selection = _normalize_selection(symbols, mix_logits, symbols_used)
    selection_str = [f'{symbol}: {ratio:3.1f}%' for ratio, symbol in selection]
    print('_selection,{}'.format(','.join(selection_str)))
    for ratio, symbol in selection:
        description = '?'
        if symbol in config_ib.SYMBOL_TO_INFO:
            description = config_ib.SYMBOL_TO_INFO[symbol]['description']
        print(f'{symbol},{ratio:3.1f}%,{description}')


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
