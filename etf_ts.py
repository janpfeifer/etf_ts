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
from typing import Dict, List, Optional, Text, Tuple

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
flags.mark_flag_as_required('data')

flags.DEFINE_integer('max_age_days', 10,
                     'Maximum number of days before updating cached data.')
flags.DEFINE_list(
    'symbols', None,
    'List of symbols to use, as opposed to reading them from InteractiveBrokers selection.')
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
    'stats', 'per_asset,greedy,average,commons,mix,selection',
    'List of stats to output. A selection of: per_asset, etc.'
)
flags.DEFINE_integer('mix_steps', 300,
                     'Number of steps to optimize each period in mixed strategy.')
flags.DEFINE_integer('commons_top_k', None, 'Only use the TopK assets of when using weighted averaged (commons)')



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

flags.DEFINE_bool(
    'include_us', False,
    'If true include US ETFs: they wont be available for Swiss residents starting on 1/1/2020')

flags.DEFINE_integer(
    'batch_size', None,
    'If set, it will try to download asset information in batches. This is useful to detect bugs '
    'in downloading.')

def main(argv):
    del argv  # Unused.

    tf_lib.config_gpu()

    asset_measures.TAX_ON_DIVIDENDS_PERCENTAGE = FLAGS.tax_on_dividends
    dense_measures.MAX_DAYS = FLAGS.max_days

    # Select and sort symbols.
    symbols = _get_symbols(FLAGS.symbols)

    # Download data or reload it from disk cache.
    dmgr = data_manager.DataManager(FLAGS.data)
    if not FLAGS.batch_size:
        symbols = dmgr.DownloadRawDataForList(
            symbols, max_age_days=FLAGS.max_age_days)
    else:
        n = FLAGS.batch_size
        batches = [
            symbols[ii * n:(ii + 1) * n] 
            for ii in range((len(symbols) + n - 1) // n )]
        symbols = []
        for ii, batch in enumerate(batches):
            print(f'\n(Down)Loading data for {batch} (batch {ii} out of {len(batches)})')
            symbols = symbols + dmgr.DownloadRawDataForList(
                batch, max_age_days=FLAGS.max_age_days)

    # Calculate independent (from each other) derived information if not loaded from cache.
    num_symbols = len(symbols)
    for ii, symbol in enumerate(symbols):
        if not asset_measures.HasDerivedValues(dmgr.data[symbol]) or FLAGS.force_recalc:
            description = config_ib.SYMBOL_TO_INFO[symbol]['description']
            logging.info(f'Calculating derived values for {symbol}: {description} - ({num_symbols - ii} missing)')
            dmgr.data[symbol] = asset_measures.AddDerivedValues(
                dmgr.data[symbol], dmgr.dividends[symbol], symbol)
            dmgr.SaveData(symbol)

    # Calculate dense ordered arrays.
    fields, mask, all_serials = dense_measures.DenseMeasureMatrices(
        dmgr.data, symbols)
    mask = tf.constant(mask, dtype=tf.bool)

    # Extra metrics calculated in Tensorflow
    fields['AdjustedLogDailyGain'] = optimizations.adjusted_log_gains(
        fields['LogDailyGain'], FLAGS.loss_cost, FLAGS.gain_power)

    # Get total assets and mask for only symbols with total_assets.
    total_assets = _get_total_assets(dmgr, symbols, mask)

    # Print out gains for each symbol.
    # Header of all outputs.
    print('Symbol,Gain,Adjusted Gain,Initial,Final,Total Assets,Description')
    if 'average' in FLAGS.stats:
        average(symbols, mask, fields, all_serials)
    if 'commons' in FLAGS.stats:
        average(symbols, mask, fields, all_serials, total_assets = total_assets)
    # if 'greedy' in FLAGS.stats:
    #     greedy(symbols, mask, fields)
    if 'mix' in FLAGS.stats:
        mix_previous_period(symbols, mask, fields, all_serials)
    if 'per_asset' in FLAGS.stats:
        per_asset_gains(symbols, mask, fields, total_assets)
    if 'selection' in FLAGS.stats:
        assets_selection(symbols, mask, fields, all_serials)


def _get_symbols(selection: Optional[List[Text]]) -> List[Text]:
    if not FLAGS.include_us:
        config_ib.exclude_us_etfs()
    symbols = config_ib.extract_ib_symbols(FLAGS.data, FLAGS.max_age_days)
    if selection is not None:
        if selection[0] == 'config':
            logging.info('Using pre-selected list of assets to choose from.')
            symbols = config.TICKERS
        elif selection[0] == 'vanguard':
            filtered = []
            for symbol in symbols:
                if config_ib.SYMBOL_TO_INFO[symbol]['description'].lower().find('vanguard') >= 0:
                    filtered.append(symbol)
            symbols = filtered
        else:
            symbols = selection
            logging.info(f'Symbols: {symbols}')
    return symbols


def _get_total_assets(dmgr: data_manager.DataManager, symbols: List[Text], 
                      mask: tf.Tensor) -> tf.Tensor:
    """Returns total assets tensor (with 0 for missing values)."""
    total_assets = []
    for symbol in symbols:
        if symbol in dmgr.total_assets and dmgr.total_assets[symbol] is not None:
            total_assets.append(float(dmgr.total_assets[symbol]))
        else:
            total_assets.append(0.0)
    return tf.constant(total_assets, dtype=tf.float32)


def per_asset_gains(symbols: List[Text], mask: tf.Tensor, fields: Dict[Text, tf.Tensor], total_assets: tf.Tensor) -> None:
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
        currency = '?'
        if symbol in config_ib.SYMBOL_TO_INFO:
            description = config_ib.SYMBOL_TO_INFO[symbol]['description']
            currency = config_ib.SYMBOL_TO_INFO[symbol]['currency']
        adjusted_gain = total_adjusted_gain[symbol_idx]

        initial_value = initial_values[symbol_idx]
        final_value = final_values[symbol_idx]
        print(f'{symbol},{gain:.4f},{adjusted_gain:.4f},{initial_value:.2f},{final_value:.2f},{total_assets[symbol_idx]:.2g},{description}')


def average(symbols: List[Text], 
            mask: np.ndarray, 
            fields: Dict[Text, tf.Tensor], 
            all_serials: List[int], 
            total_assets: Optional[tf.Tensor] = None) -> None:
    log_gains = fields['LogDailyGain']
    log_gains = tf.where(mask, log_gains, tf.zeros_like(
        log_gains, dtype=tf.float32))

    all_gains: tf.Tensor = tf.constant(0.0)
    all_adjusted_gains: tf.Tensor = tf.constant(0.0)

    if total_assets is not None:
        if tf.reduce_sum(total_assets) == 0:
            raise ValueError('Invalid weighted average: sum of total_assets is 0!')
        if FLAGS.commons_top_k is not None and len(total_assets) > FLAGS.commons_top_k:
            k = FLAGS.commons_top_k
            top_k, top_k_indices = tf.math.top_k(total_assets, k)
            if tf.reduce_min(top_k).numpy() > 0:   # If there are 0s in top-k, there is no selection to make.
                sum_all_assets =tf.reduce_sum(total_assets).numpy()
                selection_pct = (tf.reduce_sum(top_k).numpy() / sum_all_assets) * 100.0
                logging.info(f'- commons_top_k (k={k}, selected {selection_pct:.2f}% of {sum_all_assets:.2g})')
                total_assets=tf.scatter_nd(
                    tf.expand_dims(top_k_indices, -1), 
                    top_k, shape=total_assets.shape)
        logits = tf.math.log1p(total_assets)
        mask = tf.math.logical_and(mask, total_assets > 0.0)

    else:
        logits = tf.zeros(dtype=tf.float32, shape=tf.shape(log_gains)[-1])

    is_last = False
    for ii in range(config.REPORT_PERIOD_YEARS):
        start_idx = -(ii + 1) * config.YEARLY_PERIOD_IN_SERIAL
        if -start_idx >= log_gains.shape[0]:
            start_idx = -log_gains.shape[0]
            is_last = True
        end_idx = -ii * config.YEARLY_PERIOD_IN_SERIAL
        if ii == 0:
            period_mask  = mask[start_idx:, :]
            period_log_gains = log_gains[start_idx:, :]
            period_serials = all_serials[start_idx:]
        else:
            period_mask = mask[start_idx:end_idx, :]
            period_log_gains = log_gains[start_idx:end_idx, :]
            period_serials = all_serials[start_idx:end_idx]

        # Find mask for logits of symbos used.
        symbols_used = tf.constant(
            dense_measures.SelectSymbolsFromMask(period_serials, period_mask),
            dtype=tf.bool)
        logging.debug(f'- period {ii}: {np.count_nonzero(symbols_used)} symbols used.')

        # Find gains for the year.
        mix_gain, adjusted_mix_gain = optimizations.mix_gain(
            logits, symbols_used, period_log_gains, FLAGS.loss_cost,
            FLAGS.gain_power)
        all_gains += tf.math.log(mix_gain)
        all_adjusted_gains += tf.math.log(adjusted_mix_gain)

        if is_last:
            break

    cycles = -start_idx
    years = float(cycles) / config.YEARLY_PERIOD_IN_SERIAL
    name = 'average' if total_assets is None else 'commons'
    all_gains, all_adjusted_gains = _report_pct_all_gains(name, all_gains, all_adjusted_gains, cycles)
    print(f'_{name} (last {years:.1f} years),' +
          f'{all_gains:.2f},{all_adjusted_gains:.2f},,,,Gains (p.a.), readjusted every year.')
    if total_assets is not None:
        assets_mix = tf_lib.masked_softmax(logits, total_assets > 0.0).numpy()
        for idx, assets in enumerate(total_assets.numpy()):
            if assets > 0:
                symbol = symbols[idx]
                description = '?'
                if symbol in config_ib.SYMBOL_TO_INFO:
                    description = config_ib.SYMBOL_TO_INFO[symbol]['description']
                print(f'\t{symbol},{assets_mix[idx]*100.0:.1f}%,{description}')


def mix_previous_period(symbols: List[Text], mask: np.ndarray, fields: Dict[Text, tf.Tensor], all_serials: List[int]) -> None:
    log_gains = tf.convert_to_tensor(fields['LogDailyGain'], dtype=tf.float32)
    hparams = {
        'steps': FLAGS.mix_steps,
        'learning_rate': 0.1,  # 3.3e-2,
        'loss_cost': FLAGS.loss_cost,
        'gain_power': FLAGS.gain_power,
        'l1': 1e-3,
        'l2': 1e-6,
    }
    all_gains: tf.Tensor = tf.constant(0.0)
    all_adjusted_gains: tf.Tensor = tf.constant(0.0)
    apply_cycles = int(config.YEARLY_PERIOD_IN_SERIAL *
                       FLAGS.mix_applying_period)

    for last_ii_year in range(config.REPORT_PERIOD // apply_cycles):
        # Identify range where to apply new mix.
        apply_start = (-last_ii_year - 1) * apply_cycles
        if apply_start + len(all_serials) <= 0:
            logging.info('Not enough data, stopping mix here.')
            apply_start += apply_cycles
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
            apply_start += apply_cycles
            break
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

        (_, _, mix_logits, mix) = (
            optimizations.optimize_mix(symbols, symbols_used, train_gains, hparams))
        selection = _normalize_selection(symbols, mix_logits, symbols_used)
        selection = [f'{symbol}: {ratio:3.1f}%' for ratio, symbol in selection]
        logging.info('  Selection of assets: {}'.format(', '.join(selection)))

        # Use mix to the "apply" period. Collect those gains.
        mix_gain, adjusted_mix_gain = optimizations.mix_gain(
            mix_logits, symbols_used, apply_gains, FLAGS.loss_cost, FLAGS.gain_power)
        all_gains += tf.math.log(mix_gain)
        all_adjusted_gains += tf.math.log(adjusted_mix_gain)

        # Report gain of period (in %age change)
        logging.debug(f'mix_gain={mix_gain:.4f}, adjusted_mix_gain={adjusted_mix_gain:.4f}, apply_cycles={apply_cycles}')
        mix_gain = optimizations.annualized_gain(mix_gain, apply_cycles)
        mix_gain = 100.0 * (mix_gain - 1.0)
        adjusted_mix_gain = optimizations.annualized_gain(
            adjusted_mix_gain, apply_cycles)
        adjusted_mix_gain = 100.0 * (adjusted_mix_gain - 1.0)
        logging.info(f'  gain={mix_gain:.2f}%, adjusted_gain={adjusted_mix_gain:.2f}% (both annualized)')

    all_gains, all_adjusted_gains = _report_pct_all_gains('mix', all_gains, all_adjusted_gains, -apply_start)
    print(f'_mix (last {config.REPORT_PERIOD_YEARS} years),' +
          f'{all_gains:.2f},{all_adjusted_gains:.2f},,,' +
          f'Trained with preceding {FLAGS.mix_training_period} year(s) ' +
          f'and then applied for {FLAGS.mix_applying_period} year(s).')


def _report_pct_all_gains(name: Text, all_gains: tf.Tensor, all_adjusted_gains: tf.Tensor, cycles: int) -> Tuple[tf.Tensor, tf.Tensor]:
    all_gains = tf.math.exp(all_gains)
    logging.info(f'\t{name}: all_gains={all_gains:.4f}, cycles={cycles}')
    all_gains = optimizations.annualized_gain(all_gains, cycles)
    all_gains = 100.0 * (all_gains - 1.0)

    all_adjusted_gains = tf.math.exp(all_adjusted_gains)
    logging.info(f'\t{name}: all_adjusted_gains={all_adjusted_gains:.4f}, cycles={cycles}')
    all_adjusted_gains = optimizations.annualized_gain(all_adjusted_gains, cycles)
    all_adjusted_gains = 100.0 * (all_adjusted_gains - 1.0)
    return all_gains, all_adjusted_gains


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
            print(f'_greedy {period_desc[period_idx]} {argmax_desc[argmax_idx]},{total:.4f},{total_adjusted:.4f},,,' +
                  f'Selecting asset with best {argmax_desc[argmax_idx]} yield of last {period_desc[period_idx]} period. ' +
                  f' Gains (p.a.) of last {config.REPORT_PERIOD_YEARS}')


if __name__ == '__main__':
    app.run(main)
