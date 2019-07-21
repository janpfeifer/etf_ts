# dense_measures create matrices for each one of the measures for all
# days considered, with zeroes for assets not available, and a final
# mask of which asset is available in each day.

from absl import logging

import math
import numpy as np
import pandas as pd
import datetime
import time
import sys

from typing import Dict, List, Set, Text, Tuple

import asset_measures
import config
import data_manager


_MAX_SERIAL = sys.maxsize


def DenseMeasureMatrices(dmgr: data_manager.DataManager, ordered_symbols: List[Text]
                         ) -> Tuple[Dict[Text, np.ndarray], np.ndarray]:
    """Returns a mapping of field name to associated data, and a bool mask of values present.

    Args:
        dmgr: DataManager containing the data for all symbols.
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
    """
    logging.info('Generating dense matrices.')

    # Finds number of rows needed: the number of individual serial numbers.
    first_serial = asset_measures.StringDateToSerial(config.START_DATE)
    serials_set = set()  # type: Set[int]
    for symbol in ordered_symbols:
        serials = set(dmgr.data[symbol]['Serial'])
        if serials_set is None:
            serials_set = serials
        else:
            serials_set = serials_set.union(serials)
    all_serials = sorted(
        filter(lambda x: x >= first_serial, list(serials_set)))

    # Initalizes matrices with zeros.
    rows = len(all_serials)
    columns = len(ordered_symbols)
    field_to_ndarray = {
        field: np.zeros(shape=[rows, columns], dtype=float) for field in config.FIELDS_FOR_TENSORFLOW
    }
    mask = np.zeros(shape=[rows, columns], dtype=bool)

    # Loop sequentially over serial numbers.
    current_indices = [0 for symbol in ordered_symbols]
    for serial_idx, serial in enumerate(all_serials):
        active_symbols = _FindActive(
            dmgr, ordered_symbols, serial, current_indices)
        if not len(active_symbols):
            raise ValueError('No active symbols for serial {} == {}'.format(
                serial, asset_measures.SerialDateToString(serial)))
        for symbol_idx in active_symbols:
            df = dmgr.data[ordered_symbols[symbol_idx]]
            mask[serial_idx, symbol_idx] = True
            for field in config.FIELDS_FOR_TENSORFLOW:
                field_to_ndarray[field][serial_idx,
                                        symbol_idx] = df[field][current_indices[symbol_idx]]
    return field_to_ndarray, mask


def _FindActive(dmgr: data_manager.DataManager, ordered_symbols: List[Text],
                target_serial: int, current_indices: List[int]) -> List[int]:
    active = []
    for symbol_idx, symbol in enumerate(ordered_symbols):
        serial_idx = current_indices[symbol_idx]
        serials = dmgr.data[symbol]['Serial']
        while serial_idx < len(serials) and serials[serial_idx] < target_serial:
            serial_idx += 1
        current_indices[symbol_idx] = serial_idx
        if serial_idx < len(serials) and serials[serial_idx] == target_serial:
            active.append(symbol_idx)
    return active
