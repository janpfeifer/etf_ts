"""Library to download hitorical trading data."""
from absl import logging
import datetime
import os
import pandas as pd
import requests
from typing import Dict
import urllib
from yahoofinancials import YahooFinancials

import config

# Create your own API Key in WorldTradingData.com and enter it in this file, under the path.
WTD_API_KEY_FILE_NAME = 'WTD_API_KEY.txt'
_PARSE_ISO_8601_FORMAT = '%Y-%m-%d %H:%M:%S.%f'


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

  def _DownloadPricesCSVFromYahooFinance(self, symbol: str) -> str:
    asset = YahooFinancials(symbol)
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    historical_data = asset.get_historical_price_data(
        start_date=config.DEFAULT_MIN_DATE, end_date=today,
        time_interval='daily')
    prices = historical_data[symbol]['prices']
    lines = ['Date,Open,Close,High,Low,Volume']  # Header
    for e in prices:
      if e['volume'] is None:
        continue
      lines.append(f'{e["formatted_date"]},{e["open"]},{e["close"]},' +
                   f'{e["high"]},{e["low"]},{e["volume"]}')
    lines.append('')  # This will add a last line break.
    csv_data = "\n".join(lines)
    return csv_data

  def _DownloadDividendsCSVFromYahooFinance(self, symbol: str) -> str:
    asset = YahooFinancials(symbol)
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    dividends = asset.get_daily_dividend_data(
        start_date=config.DEFAULT_MIN_DATE, end_date=today)[symbol]
    lines = ['Date,Amount']  # Header
    for e in dividends:
      if e['amount'] is None:
        continue
      lines.append(f'{e["formatted_date"]},{e["amount"]}')
    lines.append('')  # This will add a last line break.
    csv_data = "\n".join(lines)
    return csv_data

  def _DownloadRawDataToFile(self, symbol: str) -> None:
    """Downloads data for the given symbol given and saves in the local directory."""
    logging.info(
        'Downloading historical price and dividend data for %s', symbol)

    # Daily prices.
    if symbol in config.TICKERS_FROM_YAHOO_FINANCE:
      csv_data = self._DownloadPricesCSVFromYahooFinance(symbol)
    else:
      csv_data = self._DownloadPricesCSVFromWorldTradingData(symbol)
    p = self.PathForRawData(symbol)
    with open(p + '/data.csv', 'w') as f:
      f.write(csv_data)

    # Dividends.
    csv_data = self._DownloadDividendsCSVFromYahooFinance(symbol)
    with open(p + '/dividends.csv', 'w') as f:
      f.write(csv_data)

    with open(p + '/timestamp.txt', 'w') as f:
      f.write('{}'.format(datetime.datetime.now()))

  def DownloadRawData(self, symbol: str, max_age_days: int = 30) -> None:
    """Downloads data for the given symbol, but reuses previous download if not < max_age_days."""
    p = self.PathForRawData(symbol)
    if not os.path.isfile(p + '/timestamp.txt') or not os.path.isfile(p + '/data.csv'):
      self._DownloadRawDataToFile(symbol)
    else:
      with open(p + '/timestamp.txt', 'r') as f:
        prev = datetime.datetime.strptime(f.read(), _PARSE_ISO_8601_FORMAT)
        if (datetime.datetime.now() - prev) > datetime.timedelta(days=max_age_days):
          self._DownloadRawDataToFile(symbol)

    # Convert CSV file to a Pandas dataframe.
    df = pd.read_csv(p + '/data.csv')
    if 'Date' not in df or len(df['Date']) == 0:
      raise ValueError(f'Failed to load data for {symbol}')
    self._data[symbol] = df
    dividends = pd.read_csv(p + '/dividends.csv')
    self._dividends[symbol] = dividends

  def SaveData(self, symbol: str) -> None:
    """Saves updated data for given symbol: this is useful to cache also dervided columns. Timestamp is unchanged."""
    if not symbol in self._data:
      raise ValueError(f'SaveData for unknown symbol "{symbol}"')
    df = self._data[symbol]
    p = self.PathForRawData(symbol)
    df.to_csv(p + '/data.csv', index=False)
