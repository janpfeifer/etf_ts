# asset_measures calculates measurements for an asset that are
# only dependent on the asset inputs themselves (and not dependent on other
# assets or general market information).
#
# Some are expensive to calculate and they can be cached by the data_manager,
# along with the downloaded original data.

import math
import numpy as np
import pandas as pd
import datetime
import time

from typing import Tuple

# Defines the largest window needed for derived data: we can use these first
# elements of the input, since the derived data will not yet be available.
MAX_WINDOW_SIZE = 21

# Last value calculated, used to check if derived values are present.
_LAST_DERIVED_VALUE_CALCULATED = 'DailyGainPctVol'


def SerialDateToString(srl_no: int) -> str:
    new_date = datetime.datetime(1970, 1, 1, 0, 0) + datetime.timedelta(srl_no)
    return new_date.strftime("%Y-%m-%d")


def StringDateToSerial(date_str: str) -> int:
    time_parts = time.strptime(date_str, "%Y-%m-%d")
    return int(round(time.mktime(time_parts) / (24 * 3600)))


def AddDerivedValues(df: pd.DataFrame) -> None:
    """Add some standard derived values to dataframe."""
    logging.info(f'Calculating derived values for {symbol}')
    df['Serial'] = df['Date'].apply(StringDateToSerial)
    df['MidOpenClose'] = (df['Open'] + df['Close']) / 2.0
    df['MidHighLow'] = (df['High'] + df['Low']) / 2.0
    df['DiffHighLow'] = df['High'] - df['Low']
    df['DiffHighLowPct'] = (100.0 * df['DiffHighLow'] / df['High']).apply(
        lambda x: min(x, 20.0))
    df['DailyGain'] = df['Close'] - df['Open']
    df['DailyGainPct'] = 100.0 * df['DailyGain'] / df['Close']
    df['DeltaSerial'] = df['Serial'] - df['Serial'].shift(+1)

    # Add volatility metrics.
    PctVolatility(df, "Close")
    Volatility(df, "DailyGainPct")


def VolatilitySubRange(values: np.array, weights: np.array) -> Tuple[float, float]:
    """Return weighted percentual volatility for a range and mean."""
    weights_sum = weights.sum()
    mean = (values * weights).sum() / weights_sum
    sq_values_diff = (values - mean)**2
    std = math.sqrt((sq_values_diff * weights).sum() / weights_sum)
    return std, mean


def PctVolatilitySubRange(values: np.array, weights: np.array) -> float:
    """Return weighted percentual volatility for a range in percentage of mean."""
    std, mean = VolatilitySubRange(values, weights)
    return 100.0 * std / mean


def PctVolatility(df: pd.DataFrame, field: str ='Close', weight_field: str ='Volume', window_size: int = MAX_WINDOW_SIZE):
    """Set field <field>+'PctVol' with percentual volatility for each ticker.

    Volatility is defined as the standard deviation of the given <field>
    value over the <window_size> days (past), weighted by <weight_field>.

    The volatility is normalized by the mean, and multiplied by 100 to give
    the "percentual volatility".

    See details in https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance
    """
    values = df[field]
    weights = df[weight_field]
    pct_volatility = [math.nan] * (window_size - 1)
    for ii in range(0, values.size - window_size + 1):
        pct_volatility.append(PctVolatilitySubRange(
            values[ii:ii + window_size], weights[ii:ii + window_size]))
    df[field + 'PctVol'] = pct_volatility


def Volatility(df: pd.DataFrame, field: str = 'DailyGainPct', weight_field: str ='Volume', window_size: int = MAX_WINDOW_SIZE):
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
