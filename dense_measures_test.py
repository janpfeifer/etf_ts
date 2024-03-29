from absl import app
from absl import logging
from absl.testing import absltest

import numpy as np
import pandas as pd
import unittest

import asset_measures
import dense_measures


class DenseMeasuresTest(absltest.TestCase):

    def setUp(self):
        data = [
            {'Date': '2019-01-01', 'Open': 10.0, 'Close': 11.0,
                'High': 11.5, 'Low': 9.9, 'Volume': 100},
            {'Date': '2019-01-02', 'Open': 12.0, 'Close': 11.0,
                'High': 11.5, 'Low': 9.9, 'Volume': 100},

            # Some data points don't have high/low and get set to 0.
            {'Date': '2019-01-07', 'Open': 10.8, 'Close': 11.0,
                'High': 0, 'Low': 0, 'Volume': 100},
            {'Date': '2019-01-08', 'Open': 11.1, 'Close': 13.0,
                'High': 11.5, 'Low': 9.9, 'Volume': 100},
        ]

        # Repeat + 9 times, with 10 days in between.
        for ii in range(9):
            for jj in range(4):
                date = data[jj]['Date']
                date = asset_measures.StringDateToSerial(date) + (ii + 1) * 10
                row = data[jj].copy()
                row['Date'] = asset_measures.SerialDateToString(date)
                data.append(row)

        data = {
            'base': pd.DataFrame(data)
        }
        data['base_p1'] = pd.DataFrame()
        data['base_p1']['Date'] = data['base']['Date']
        data['base_p2'] = pd.DataFrame()
        data['base_p2']['Date'] = data['base']['Date']
        for field in ['Open', 'Close', 'High', 'Low', 'Volume']:
            data['base_p1'][field] = data['base'][field] + 1.0
            data['base_p2'][field] = data['base'][field] + 2.0

        dividends = pd.DataFrame([
            # On day that asset is negotiated.
            {'Date': '2019-01-01', 'Amount': 0.3},
            {'Date': '2019-01-06', 'Amount': 0.5},  # On off-day.
        ])

        for symbol in data.keys():
            data[symbol] = asset_measures.AddDerivedValues(
                data[symbol], dividends, symbol)

        # We need more data to calculate volatility,
        # but after that, for our test, we only need a few rows.
        data['base'] = data['base'][:12]
        data['base_p1'] = data['base_p1'][:11]
        data['base_p2'] = data['base_p2'].iloc[[0, 1, 2, 3, 8, 9, 10, 11]]
        for symbol in data.keys():
            data[symbol] = data[symbol].reset_index(drop=True)
        self.data_ = data

    def test_DenseMeasureMatrices(self):
        dense_measures.MAX_DAYS = None
        symbols = sorted(list(self.data_.keys()))
        logging.debug(f'Inputs: {symbols}')
        for symbol in symbols:
            logging.debug(f'\n  {symbol}:')
            logging.debug(self.data_[symbol])

        dense, mask, serials = dense_measures.DenseMeasureMatrices(
            self.data_, symbols)

        logging.debug(f'mask={mask}')
        mask = np.array(mask, np.bool)
        self.assertTrue(np.all(mask[:, 0]),
                        'base dataset must all be included.')
        self.assertTrue(np.all(mask[:11, 1]),
                        'base_p1 dataset, only first 11 are included.')
        self.assertTrue(np.all(~mask[11:, 1]),
                        'base_p1 dataset, only first 11 are included.')
        self.assertTrue(np.all(mask[:4, 2]),
                        'base_p2 dataset has values for 0:4 and 8:12')
        self.assertTrue(np.all(~mask[4:8, 2]),
                        'base_p2 dataset has values for 0:4 and 8:12')
        self.assertTrue(np.all(mask[8:, 2]),
                        'base_p2 dataset has values for 0:4 and 8:12')

        # print(f'dense={dense}')
        for field in dense.keys():
            self.assertEqual(dense[field].shape, (12, 3))
            self.assertFalse(np.any(np.isnan(dense[field])))

    def test_SelectSymbolsFromMask(self):
        dense_measures.MAX_ACCEPTABLE_SKIP = 7
        serials = np.array([10, 11, 14, 18, 19])
        mask = np.array([
            [True, False, True],
            [True, False, True],
            [True, True, False],
            [True, True, False],
            [False, True, True],
        ], dtype=bool)
        select_from = dense_measures.SelectSymbolsFromMask(serials, mask)

        # Column 2 skips from serial 11 to 19, that is > 7, hence should be false.
        # Others should be all selected.
        self.assertEqual(select_from, [True, True, False])


if __name__ == '__main__':
    absltest.main()
