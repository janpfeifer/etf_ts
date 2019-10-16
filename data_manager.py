"""Library to download hitorical trading data."""
from absl import logging
import datetime
import os
import pandas as pd
import requests
from typing import Dict, List, Optional, Text, Union
import urllib
from yahoofinancials import YahooFinancials

import config

# Create your own API Key in WorldTradingData.com and enter it in this file, under the path.
WTD_API_KEY_FILE_NAME = 'WTD_API_KEY.txt'
_PARSE_ISO_8601_FORMAT = '%Y-%m-%d %H:%M:%S.%f'

MIN_NUM_PRICE_VALUES = config.YEARLY_PERIOD_IN_SERIAL / 2


class DataManager:
  """Manages download and cache of historical ticker data, using WorldTradingData.com"""

  def __init__(self, base_path: str):
    """Constructor, takes a WorldTradingData key and a base path where to store cached data."""
    self._base = base_path
    self._data: Dict[str, pd.DataFrame] = dict()
    self._dividends: Dict[str, pd.DataFrame] = dict()
    with open('{}/{}'.format(base_path, WTD_API_KEY_FILE_NAME), 'r') as file:
      self._wtd_key = file.read().replace('\n', '')

  @property
  def base(self):
    """Returns base path used for storage."""
    return self._base

  @property
  def data(self):
    """Returns a dictionary of symobl name to Pandas DataFrame with all data downloaded so far."""
    return self._data

  @property
  def dividends(self):
    """Returns a dictionary of symobl name to Pandas DataFrame with dividends information."""
    return self._dividends

  def PathForRawData(self, symbol: str) -> str:
    """Creates returns directory for symbol."""
    p = '{}/WTD/{}'.format(self.base, symbol)
    if not os.path.exists(p):
      os.makedirs(p)
    return p

  def _DownloadPricesCSVFromWorldTradingData(self, symbol: str) -> str:
    url = 'https://www.worldtradingdata.com/api/v1/history?symbol={}&output=csv&sort=oldest&api_token={}'.format(
        symbol, self._wtd_key)
    response = requests.get(url)
    return response.text

  def _DownloadPricesCSVFromYahooFinance(self, symbols: List[Text]) -> List[Optional[Text]]:
    log_symbols('Loading prices from Yahoo', symbols)
    asset = YahooFinancials(symbols)
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    historical_data = asset.get_historical_price_data(
        start_date=config.DEFAULT_MIN_DATE, end_date=today,
        time_interval='daily')

    csv_data = []
    for symbol in symbols:
      if symbol not in historical_data or historical_data[symbol] is None or 'prices' not in historical_data[symbol]:
        csv_data.append(None)
        logging.info('No price information for symbol "{}"'.format(symbol))
        continue
      prices = historical_data[symbol]['prices']
      lines = ['Date,Open,Close,High,Low,Volume']  # Header
      for e in prices:
        if e['volume'] is None:
          continue
        lines.append(f'{e["formatted_date"]},{e["open"]},{e["close"]},' +
                     f'{e["high"]},{e["low"]},{e["volume"]}')
      if len(lines) <= 1:
        csv_data.append(None)
        logging.info(
            'Price information for symbol "{}" is empty'.format(symbol))
        continue
      lines.append('')  # This will add a last line break.
      csv_data.append('\n'.join(lines))
    return csv_data

  def _DownloadDividendsCSVFromYahooFinance(self, symbols: List[Text]) -> List[Optional[Text]]:
    log_symbols('Loading dividends', symbols)
    csv_data = []
    for symbol in symbols:
      asset = YahooFinancials(symbol)
      today = datetime.datetime.now().strftime("%Y-%m-%d")
      dividends = asset.get_daily_dividend_data(
          start_date=config.DEFAULT_MIN_DATE, end_date=today)[symbol]
      if dividends is None:
        csv_data.append(None)
        continue
      lines = ['Date,Amount']  # Header
      for e in dividends:
        if e['amount'] is None:
          continue
        lines.append(f'{e["formatted_date"]},{e["amount"]}')
      lines.append('')  # This will add a last line break.
      csv_data.append("\n".join(lines))
    return csv_data

  def _FromYahoo(self, symbol: Text) -> bool:
    return symbol in config.TICKERS_FROM_YAHOO_FINANCE or symbol.endswith('.SW') or symbol.endswith('.L')

  def _DownloadRawDataToFile(self, symbol: str) -> None:
    """Downloads data for the given symbol given and saves in the local directory."""
    logging.info(
        'Downloading historical price and dividend data for %s', symbol)

    # Daily prices.
    if self._FromYahoo(symbol):
      csv_data = self._DownloadPricesCSVFromYahooFinance([symbol])[0]
      if csv_data is None:
        raise ValueError('No data for "{}"'.format(symbol))
    else:
      csv_data = self._DownloadPricesCSVFromWorldTradingData(symbol)
    p = self.PathForRawData(symbol)
    with open(p + '/data.csv', 'w') as f:
      f.write(csv_data)

    # Dividends.
    csv_data = self._DownloadDividendsCSVFromYahooFinance([symbol])[0]
    if csv_data is not None:
      with open(p + '/dividends.csv', 'w') as f:
        f.write(csv_data)

    with open(p + '/timestamp.txt', 'w') as f:
      f.write('{}'.format(datetime.datetime.now()))

  def _LoadData(self, symbol: Text) -> bool:
    """Loads data for symbol, returns `False` if not enough data worth using."""
    # Convert CSV file to a Pandas dataframe.
    p = self.PathForRawData(symbol)
    df = pd.read_csv(p + '/data.csv')
    if 'Date' not in df or len(df['Date']) == 0:
      raise ValueError(f'Failed to load data for {symbol}')
    if len(df.index) < config.YEARLY_PERIOD_IN_SERIAL:
      return False
    self._data[symbol] = df
    if os.path.isfile(p + '/dividends.csv'):
      dividends = pd.read_csv(p + '/dividends.csv')
    else:
      dividends = None
    self._dividends[symbol] = dividends
    return True

  def DownloadRawData(self, symbol: Text, max_age_days: int=30) -> bool:
    """Downloads data for the given symbol, but reuses previous download if not < max_age_days."""
    p = self.PathForRawData(symbol)
    if not os.path.isfile(p + '/timestamp.txt') or not os.path.isfile(p + '/data.csv'):
      self._DownloadRawDataToFile(symbol)
    else:
      with open(p + '/timestamp.txt', 'r') as f:
        prev = datetime.datetime.strptime(f.read(), _PARSE_ISO_8601_FORMAT)
        if (datetime.datetime.now() - prev) > datetime.timedelta(days=max_age_days):
          self._DownloadRawDataToFile(symbol)
    self._LoadData(symbol)

  def DownloadRawDataForList(self, symbols: List[Text], max_age_days: int=30) -> List[Text]:
    """Downloads data for the given symbol, but reuses previous download if not < max_age_days.

    Args:
      symbols: list of symbols to download data for.

    Returns:
      List of symbols with enough data.
    """
    log_symbols('Loading data', symbols)

    # Finds symbols that need downloading.
    need_downloading = []
    for symbol in symbols:
      p = self.PathForRawData(symbol)
      if not os.path.isfile(p + '/timestamp.txt') or not os.path.isfile(p + '/data.csv'):
        need_downloading.append(symbol)
      else:
        with open(p + '/timestamp.txt', 'r') as f:
          prev = datetime.datetime.strptime(f.read(), _PARSE_ISO_8601_FORMAT)
          if (datetime.datetime.now() - prev) > datetime.timedelta(days=max_age_days):
            need_downloading.append(symbol)

    # Separate symbols by source.
    yahoo_symbols = []
    wtd_symbols = []
    failed_symbols = set()
    for symbol in need_downloading:
      if self._FromYahoo(symbol):
        yahoo_symbols.append(symbol)
      else:
        wtd_symbols.append(symbol)

    # Download the Yahoo based prices.
    if yahoo_symbols:
      all_csv = self._DownloadPricesCSVFromYahooFinance(yahoo_symbols)
      for (ii, symbol) in enumerate(yahoo_symbols):
        csv = all_csv[ii]
        if csv is None:
          failed_symbols.add(symbol)
        else:
          logging.info('  Saving princing data for {}'.format(symbol))
          p = self.PathForRawData(symbol)
          with open(p + '/data.csv', 'w') as f:
            f.write(csv)

    if wtd_symbols:
      log_symbols('Loading data from WTD', wtd_symbols)
      for symbol in wtd_symbols:
        csv = self._DownloadPricesCSVFromWorldTradingData(symbol)
        if csv is None:
          failed_symbols.add(symbol)
        else:
          logging.info('  Saving princing data for {}'.format(symbol))
          p = self.PathForRawData(symbol)
          with open(p + '/data.csv', 'w') as f:
            f.write(csv)

    # Download dividends.
    need_downloading = [
        symbol for symbol in need_downloading if symbol not in failed_symbols]
    all_csv = self._DownloadDividendsCSVFromYahooFinance(need_downloading)
    for (ii, symbol) in enumerate(need_downloading):
      csv = all_csv[ii]
      if csv is not None:
        p = self.PathForRawData(symbol)
        with open(p + '/dividends.csv', 'w') as f:
          f.write(csv)

    # Save timestamp.
    for symbol in need_downloading:
      if symbol in failed_symbols:
        continue
      p = self.PathForRawData(symbol)
      with open(p + '/timestamp.txt', 'w') as f:
        f.write('{}'.format(datetime.datetime.now()))

    # Raise error if there were any failed symbols.
    if failed_symbols:
      raise ValueError('Failed to download information for symbols: {}'.format(
          ', '.join(failed_symbols)))

    # Load all the data into dataframes.
    good_symbols = []
    dropped_symbols = []
    for symbol in symbols:
      if self._LoadData(symbol):
        good_symbols.append(symbol)
      else:
        dropped_symbols.append(symbol)
    log_symbols('Dropped info due to lack of data', dropped_symbols)
    return good_symbols

  def SaveData(self, symbol: str) -> None:
    """Saves updated data for given symbol: this is useful to cache also dervided columns. Timestamp is unchanged."""
    if not symbol in self._data:
      raise ValueError(f'SaveData for unknown symbol "{symbol}"')
    df = self._data[symbol]
    p = self.PathForRawData(symbol)
    df.to_csv(p + '/data.csv', index=False)


def log_symbols(msg, symbols):
  logging.info('{} for {} symbols ([{}{}])'.format(
      msg, len(symbols), ', '.join(symbols[:5]),
      ', ...' if len(symbols) > 5 else ''))
