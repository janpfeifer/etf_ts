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
    'stats', 'per_asset',
    'List of stats to output. A selection of: per_asset, etc.'
)
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
        dmgr.DownloadRawData(symbol)

    # Calculate independent (from each other) derived information if not loaded from cache.
    for symbol in symbols:
        df = dmgr.data[symbol]
        if not asset_measures.HasDerivedValues(df) or FLAGS.force_recalc:
            logging.info(f'Calculating derived values for {symbol}')
            asset_measures.AddDerivedValues(df)
            dmgr.SaveData(symbol)

    # Calculate dense ordered arrays.
    fields, mask = dense_measures.DenseMeasureMatrices(dmgr, symbols)
    for key, values in fields.items():
        print(f' Field {key}: {values.shape}')
    print(f'Mask: {mask.shape}')

    # Convert dense arrays to transposed tensors.
    mask = tf.transpose(tf.constant(mask, dtype=tf.bool))
    for field in fields.keys():
        fields[field] = tf.transpose(
            tf.constant(fields[field], dtype=tf.float32))

    # Print out gains for each symbol.
    if 'per_asset' in FLAGS.stats:
        per_asset_gains(symbols, mask, fields)


def per_asset_gains(symbols: List[Text], mask: tf.Tensor, fields: Dict[Text, tf.Tensor]) -> None:
    """Print basic assets data (CSV) and per assets gains and adjusted gains."""
    open_ = fields['Open']
    close = fields['Open']
    initial_values = tf_lib.masked_first(open_, mask)
    final_values = tf_lib.masked_last(close, mask)
    log_gains = fields['LogDailyGain']
    log_adjusted_gains = optimizations.adjust_log_daily_gain(
        log_gains, FLAGS.loss_cost)
    total_gains = optimizations.total_gain_from_log_gains(log_gains)
    total_adjusted_gains = optimizations.total_gain_from_log_gains(
        log_adjusted_gains)

    print('symbol,gain,adjusted,initial,final')
    for symbol_idx, symbol in enumerate(symbols):
        gain = total_gains[symbol_idx]
        adjusted_gain = total_adjusted_gains[symbol_idx]
        initial_value = initial_values[symbol_idx]
        final_value = final_values[symbol_idx]
        print(f'{symbol},{gain:.4f},{adjusted_gain:.4f},{initial_value:.2f},{final_value:.2f}')


def oracle(symbols: List[Text], mask: tf.Tensor, fields: Dict[Text, tf.Tensor]) -> None:
    pass


if __name__ == '__main__':
    app.run(main)
