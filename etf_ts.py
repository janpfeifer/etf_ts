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
