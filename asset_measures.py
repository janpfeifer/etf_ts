# asset_measures calculates measurements for an asset that are
# only dependent on the asset inputs themselves (and not dependent on other
# assets or general market information).
#
# Some are expensive to calculate and they can be cached by the data_manager,
# along with the downloaded original data.

from absl import logging
import math
import numpy as np
import pandas as pd
import datetime
import time
from typing import Optional, Tuple

import config

# Defines the largest window needed for derived data: we can use these first
# elements of the input, since the derived data will not yet be available.
MAX_WINDOW_SIZE = 21

# Last value calculated, used to check if derived values are present.
#_LAST_DERIVED_VALUE_CALCULATED = 'PctDailyGainVol'
_LAST_DERIVED_VALUE_CALCULATED = 'Dividends'

# Percentage of dividends lost through taxes.
TAX_ON_DIVIDENDS_PERCENTAGE = 35.0


def SerialDateToString(srl_no: int) -> str:
    new_date = datetime.datetime(1970, 1, 1, 0, 0) + datetime.timedelta(srl_no)
    return new_date.strftime("%Y-%m-%d")


def StringDateToSerial(date_str: str) -> int:
    time_parts = time.strptime(date_str, "%Y-%m-%d")
    return int(round(time.mktime(time_parts) / (24 * 3600)))


def _trim_before_serial(df: pd.DataFrame, min_serial: int) -> pd.DataFrame:
    """Remove all entries that preceed serial value."""
    serials = df['Serial']
    for trim_start in range(len(serials)):
        if serials[trim_start] >= min_serial:
            break
    if trim_start == 0:
        return df
    df = df.loc[trim_start:].copy()
    df.index = np.arange(len(df['Serial']))
    return df


def _merge_dividends(df: pd.DataFrame, dividends: Optional[pd.DataFrame], symbol: str) -> pd.DataFrame:
    if dividends is None:
        df['Dividends'] = np.zeros_like(df['Close'], dtype=np.float)
        df['SpreadDividends'] = np.zeros_like(df['Close'], dtype=np.float)
        return df

    dividends['Serial'] = dividends['Date'].apply(StringDateToSerial)
    dividends = dividends.drop(columns='Date')
    dividends = dividends.rename(columns={'Amount': 'Dividends'})
    if 'Dividends' in df:
        df = df.drop(columns='Dividends')
    merged_df = pd.merge(df, dividends, on='Serial', how='outer')

    if merged_df['Close'].isna().any():
        # Move date of dividends to the previous market day.
        logging.info(f'Asset {symbol} has dividend paid on day without an Open value, then the merge is slower.')
        valid_serials = set(df['Serial'].tolist())
        new_dividends = []
        for _, row in dividends.iterrows():
            serial = row['Serial']
            amount = row['Dividends']
            if serial in valid_serials:
                new_dividends.append({'Serial': serial, 'Dividends': amount})
            else:
                # Search for a replacind serial for up to a month before.
                date = SerialDateToString(int(serial))
                new_serial = -1
                for ii in range(config.MONTHLY_PERIOD_IN_SERIAL):
                    new_serial = int(serial - ii - 1)
                    new_date = SerialDateToString(new_serial)
                    if new_serial in valid_serials:
                        new_dividends.append(
                            {'Serial': new_serial, 'Dividends': amount})
                        logging.info(f'Missing marked for dividend on {date} (serial={serial}), moved to {new_date} instead')
                        break
                if new_serial == -1:
                    raise ValueError(f'Missing marked for dividend on {date} (serial={serial}), no alternative date found.')
        if new_dividends:
            dividends = pd.DataFrame(new_dividends)
            merged_df = pd.merge(df, dividends, on='Serial', how='outer')
        else:
            df['Dividends'] = np.zeros_like(df['Close'], dtype=np.float)
            df['SpreadDividends'] = np.zeros_like(df['Close'], dtype=np.float)
            return df

    # Spread
    df = merged_df
    df['Dividends'] = np.where(df['Dividends'].isna(), 0.0, df['Dividends'])
    dividends = df['Dividends']
    spread = np.zeros_like(dividends, dtype=np.float)
    for idx in range(dividends.size):
        if dividends[idx] > 0.0:
            fraction = float(dividends[idx]) / float(config.SPREAD_DIVIDENDS)
            spread[min(0, idx - config.SPREAD_DIVIDENDS + 1)                   : idx + 1] += fraction
            if idx > dividends.size - config.SPREAD_DIVIDENDS:
                # Assume similar dividends will be paid next cycle: likely a better
                # assumption than no dividend will be paid.
                spread[idx + 1:] += fraction
    df['SpreadDividends'] = spread * \
        (1.0 - TAX_ON_DIVIDENDS_PERCENTAGE / 100.0)
    return df


def AddDerivedValues(df: pd.DataFrame, dividends: Optional[pd.DataFrame], symbol: str) -> pd.DataFrame:
    """Add some standard derived values to dataframe."""
    df['Serial'] = df['Date'].apply(StringDateToSerial)
    df = _merge_dividends(df, dividends, symbol)

    # Fix broken / missing dates anda data.
    if symbol in config.FIX_MIN_DATE:
        min_serial = StringDateToSerial(config.FIX_MIN_DATE[symbol])
        df = _trim_before_serial(df, min_serial)
    if symbol in config.FIX_MISSING_OPEN:
        df['Open'] = np.concatenate([[df['Close'][0]], df['Close'][:-1]])
        df['Volume'] = np.ones_like(df['Close'])
        df['High'] = np.maximum(df['Open'], df['Close'])
        df['Low'] = np.minimum(df['Open'], df['Close'])

    df['MidOpenClose'] = (df['Open'] + df['Close']) / 2.0
    df['MidHighLow'] = (df['High'] + df['Low']) / 2.0
    df['DiffHighLow'] = df['High'] - df['Low']
    df['DiffHighLowPct'] = (100.0 * df['DiffHighLow'] / df['High']).apply(
        lambda x: 0.0 if math.isnan(x) else min(x, 20.0))
    _open = df['Open'].values
    close = df['Close'].values
    next_open = np.roll(_open, -1)
    next_open[-1] = close[-1]
    daily_gain = next_open - _open
    df['DailyValuation'] = daily_gain
    logging.info('Sum Valuation={}'.format(np.sum(daily_gain)))
    daily_gain = daily_gain + df['SpreadDividends']
    logging.info('Sum Gains={}'.format(np.sum(daily_gain)))
    df['DailyGain'] = daily_gain
    df['PctDailyGain'] = 100.0 * daily_gain / _open
    df['LogDailyGain'] = np.log((_open + daily_gain) / _open)
    df['DeltaSerial'] = df['Serial'] - df['Serial'].shift(+1)
    MeanValue(df, 'Volume')

    # Add volatility metrics.
    PctVolatility(df, 'Close')
    Volatility(df, 'PctDailyGain')
    return df


def MeanValue(df: pd.DataFrame, field: str, window_size: int = MAX_WINDOW_SIZE):
    values = df[field]
    means = np.zeros_like(values)
    for idx in range(values.size):
        idx_start = max(0, idx - window_size + 1)
        means[idx] = np.mean(values[idx_start:idx + 1])
    df[f'{field}Mean'] = means


def VolatilitySubRange(values: np.array, weights: np.array) -> Tuple[float, float]:
    """Return weighted percentual volatility for a range and mean."""
    weights_sum = weights.sum()
    if weights_sum <= 0:
        weights_sum = 1.0
    mean = (values * weights).sum() / weights_sum
    sq_values_diff = (values - mean)**2
    std = math.sqrt((sq_values_diff * weights).sum() / weights_sum)
    return std, mean


def PctVolatilitySubRange(values: np.array, weights: np.array) -> float:
    """Return weighted percentual volatility for a range in percentage of mean."""
    std, mean = VolatilitySubRange(values, weights)
    if mean <= 0:
        return 0.0
    return 100.0 * std / mean


def PctVolatility(df: pd.DataFrame, field: str ='Close', weight_field: str ='Volume', window_size: int = MAX_WINDOW_SIZE):
    """Set field <field>+'PctVol' with percentual volatility for each ticker.

    Volatility is defined as the standard deviation of the given <field>
    value over the <window_size> days (past), weighted by <weight_field>.

    The volatility is normalized by the mean, and multiplied by 100 to give
    the "percentual volatility".

    # Weighted_sample_variance
    See details in https://en.wikipedia.org/wiki/Weighted_arithmetic_mean
    """
    values = df[field]
    weights = df[weight_field]
    pct_volatility = [math.nan] * (window_size - 1)
    for ii in range(0, values.size - window_size + 1):
        pct_volatility.append(PctVolatilitySubRange(
            values[ii:ii + window_size], weights[ii:ii + window_size]))
    df[field + 'PctVol'] = pct_volatility


def Volatility(df: pd.DataFrame, field: str = 'PctDailyGain', weight_field: str ='Volume', window_size: int = MAX_WINDOW_SIZE):
    """Set field <field>+'Vol' with percentual volatility for each ticker."""
    values = df[field]
    weights = df[weight_field]
    volatility = [math.nan] * (window_size - 1)
    for ii in range(0, values.size - window_size + 1):
        v, _ = VolatilitySubRange(
            values[ii:ii + window_size], weights[ii:ii + window_size])
        volatility.append(v)
    df[field + 'Vol'] = volatility


def HasDerivedValues(df: pd.DataFrame) -> bool:
    """Returns whether the dataframe has the derived values."""
    return _LAST_DERIVED_VALUE_CALCULATED in df
