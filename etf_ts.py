#!/usr/bin/env python

# etf_tf.py will train a model or using a trained model output a prediction
# of best mixture of ETFs to maximize gains, minimize volatility and risk,
# according to parameters in config.py.

# import tensorflow as tf
# from tensorflow import keras

from absl import app
from absl import flags
from absl import logging

import config
import data_manager
import asset_measures

# Flags
FLAGS = flags.FLAGS
flags.DEFINE_string(
    'data', None,
    'Base path where to store historical marked data. Can be shared across models')
flags.DEFINE_list(
    'symbols', None,
    'List of sBase path where to store historical marked data. Can be shared across models')

flags.mark_flag_as_required('data')


def main(argv):
    del argv  # Unused.

    # Download data or reload it from disk cache.
    dmgr = data_manager.DataManager(FLAGS.data)
    symbols = FLAGS.symbols
    if symbols is None:
        symbols = config.TICKERS
    for symbol in symbols:
        dmgr.DownloadRawData(symbol)

    # Calculate independent (from each other) derived information if not loaded from cache.
    for symbol in symbols:
        df = dmgr.data[symbol]
        if not asset_measures.HasDerivedValues(df):
            asset_measures.AddDerivedValues(df)
            dmgr.SaveData(symbol)


if __name__ == '__main__':
    app.run(main)
