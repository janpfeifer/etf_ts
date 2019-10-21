# dense_measures create matrices for each one of the measures for all
# days considered, with zeroes for assets not available, and a final
# mask of which asset is available in each day.

from absl import flags
from absl import logging

import math
import numpy as np
import pandas as pd
import datetime
import time
import sys

from typing import Dict, List, Set, Text, Tuple

import asset_measures
import data_manager
import config

FLAGS = flags.FLAGS

# If set, take only the latest fiven number of days.
MAX_DAYS = None

# Any skips in the data for an asset, and the asset is masked out.
MAX_ACCEPTABLE_SKIP = 15

_MAX_SERIAL = sys.maxsize


def DenseMeasureMatrices(data: Dict[Text, pd.DataFrame], ordered_symbols: List[Text]
                         ) -> Tuple[Dict[Text, np.ndarray], np.ndarray, List[int]]:
    """Returns a mapping of field name to associated data, and a bool mask of values present.

    Args:
        data: Map of asset's symbol name to dataframe with the asset's information.
        ordered_symbols: List of symbols, its order will determine the order of
          the values returned.

    Returns:
        field_to_ndarray: For each field in config.FIELDS_FOR_TENSORFLOW, create a
          matrix of shape [NUM_SERIALS, NUM_SYMBOLS] with the values for that field
          in each "serial" (serialized date, so one per day) present in the dataset
          (so weekends are skiped). If a symbol (asset) doesn't have information in
          a particular day (assets negotiated in London are shifted one day), they
          are set to 0, and the mask is set to false accordingly.
        mask: A matrix of bools, of shape [NUM_SERIALS, NUM_SYMBOLS]. It indicates
          whether information for the particular symbol is available in a particular
          "serial" (a serial number representing a day).
        used_serials: The `Serial` (date converted to serial number) values used.
    """
    logging.info('Generating dense matrices.')

    # Check for nans
    logging.info('  - checking for NaNs.')
    has_nans = False
    for symbol in ordered_symbols:
        df = data[symbol]
        for field in config.FIELDS_FOR_TENSORFLOW:
            values = df[field].values[asset_measures.MAX_WINDOW_SIZE:]
            if MAX_DAYS is not None:
                values = values[-MAX_DAYS:]
            if np.any(np.isnan(values)):
                logging.info(f'Found nan in asset {symbol}, field {field}, in rows {np.where(np.isnan(df[field].values))}')
                has_nans = True
    if has_nans:
        raise ValueError('Nan values in assets information.')

    # Finds number of rows needed: the number of individual serial numbers.
    logging.info('  - find all date serial numbers we have information.')
    first_serial = asset_measures.StringDateToSerial(config.START_DATE)
    serials_set = set()  # type: Set[int]
    for symbol in ordered_symbols:
        serials = set(data[symbol]['Serial'])
        if serials_set is None:
            serials_set = serials
        else:
            serials_set = serials_set.union(serials)
    all_serials = sorted(
        filter(lambda x: x >= first_serial, list(serials_set)))
    if MAX_DAYS is not None:
        all_serials = all_serials[-MAX_DAYS:]
    max_serial = all_serials[-1]
    max_skip = _max_skip(np.array(all_serials), max_serial)
    logging.debug(f'Combined dataset from all assets max_skip={max_skip}')

    # Trim DataFrames to only the entries that we are interest in.
    logging.info('  - limit only the days that matter and check max_skip of each symbol.')
    limited_data = {}
    disabled_symbols = set()
    for symbol in ordered_symbols:
        df = data[symbol]
        df = df[df['Serial'].isin(all_serials)].reset_index(drop=True)
        sym_max_skip = _max_skip(df['Serial'], max_serial)
        if sym_max_skip > MAX_ACCEPTABLE_SKIP:
            disabled_symbols.add(symbol)
        print(f'{symbol}: max skip = {sym_max_skip} (max_skip={max_skip})')
        limited_data[symbol] = df

    # TODO: join by key 'Serial', one symbol at a time.
    all_serials_df = pd.DataFrame({'Serial': all_serials})
    for symbol in ordered_symbols:
        limited_data[symbol] = pd.merge(
            all_serials_df, limited_data[symbol], on='Serial', how='outer')
        print(f'limited_data[{symbol}]={limited_data[symbol]}\n')

    # Initalizes matrices with zeros.
    logging.info(
        '  - zero initialize arrays for fields {}'.format(', '.join(config.FIELDS_FOR_TENSORFLOW)))
    rows = len(all_serials)
    columns = len(ordered_symbols)
    field_to_ndarray = {
        field: np.zeros(shape=[rows, columns], dtype=float) for field in config.FIELDS_FOR_TENSORFLOW
    }
    mask = np.zeros(shape=[rows, columns], dtype=bool)

    # Loop sequentially over serial numbers.
    logging.info(
        '  - loop over serials vs find all date serial numbers we have information.')
    current_indices = [0 for symbol in ordered_symbols]
    print_count = 10
    total_count = len(all_serials)
    for serial_idx, serial in enumerate(all_serials):
        if serial_idx > 0 and serial_idx % print_count == 0:
            logging.info(f'    done {serial_idx} out of {total_count}')
            if print_count * 10 < total_count:
                print_count *= 10

        active_symbols = _FindActive(
            limited_data, ordered_symbols, serial, current_indices)
        if not len(active_symbols):
            raise ValueError('No active symbols for serial {} == {}'.format(
                serial, asset_measures.SerialDateToString(serial)))
        for symbol_idx in active_symbols:
            df = limited_data[ordered_symbols[symbol_idx]]
            mask[serial_idx, symbol_idx] = True
            for field in config.FIELDS_FOR_TENSORFLOW:
                field_to_ndarray[field][serial_idx,
                                        symbol_idx] = df[field][current_indices[symbol_idx]]


    print(f'  - disable symbols with windows > {MAX_ACCEPTABLE_SKIP} days without data', list(disabled_symbols))
    for ii, symbol in enumerate(ordered_symbols):
        if symbol in disabled_symbols:
            mask[:, ii] = False

    return field_to_ndarray, mask, all_serials


def _max_skip(serials: np.ndarray, last_serial:int) -> int:
    next = np.roll(serials, -1)
    next[-1] = last_serial
    diff = next - serials
    return np.amax(diff)


def _FindActive(data: Dict[Text, pd.DataFrame], ordered_symbols: List[Text],
                target_serial: int, current_indices: List[int]) -> List[int]:
    active = []
    for symbol_idx, symbol in enumerate(ordered_symbols):
        serial_idx = current_indices[symbol_idx]
        serials = data[symbol]['Serial']
        while serial_idx < len(serials) and serials[serial_idx] < target_serial:
            serial_idx += 1
        current_indices[symbol_idx] = serial_idx
        if serial_idx < len(serials) and serials[serial_idx] == target_serial:
            active.append(symbol_idx)
    return active
