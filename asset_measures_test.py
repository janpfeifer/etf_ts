import numpy as np
import pandas as pd
import unittest

import asset_measures


class AssetMeasuresTest(unittest.TestCase):

    def setUp(self):
        data = [
            { 'Date': '2019-01-01', 'Open': 10.0, 'Close': 11.0, 'High': 11.5, 'Low': 9.9, 'Volume': 100 },
            { 'Date': '2019-01-02', 'Open': 12.0, 'Close': 11.0, 'High': 11.5, 'Low': 9.9, 'Volume': 100 },

            # Some data points don't have high/low and get set to 0.
            { 'Date': '2019-01-07', 'Open': 10.8, 'Close': 11.0, 'High': 0, 'Low': 0, 'Volume': 100 },
            { 'Date': '2019-01-08', 'Open': 11.1, 'Close': 13.0, 'High': 11.5, 'Low': 9.9, 'Volume': 100 },
        ]

        # Repeat + 9 times, with 10 days in between.
        for ii in range(9):
            for jj in range(4):
                date = data[jj]['Date']
                date = asset_measures.StringDateToSerial(date) + (ii+1)*10
                row = data[jj].copy()
                row['Date'] = asset_measures.SerialDateToString(date)
                data.append(row)
        self.df_ = pd.DataFrame(data)

        self.dividends_ = pd.DataFrame([
            { 'Date': '2019-01-01', 'Amount': 0.3 },  # On day that asset is negotiated.
            { 'Date': '2019-01-05', 'Amount': 0.4 },  # On off-day.
            { 'Date': '2019-01-06', 'Amount': 0.1 },  # On off-day again.
        ])

    def test_mix_gain(self):
        print('Inputs: open/close')
        print(self.df_.loc[:4])
        print('\nInputs: dividends')
        print(self.dividends_)

        measures = asset_measures.AddDerivedValues(self.df_, self.dividends_, 'TEST')
        print('\nOutput: derived data')
        print(measures.loc[:4])

        self.assertTrue(np.allclose(measures['DailyValuation'][:4].values, [2.0, -1.2, 0.3, -1.1]))
        self.assertTrue(np.allclose(measures['DailyGain'][:4].values, [2.3, -0.7, 0.3, -1.1]))


if __name__ == '__main__':
    unittest.main()
