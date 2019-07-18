"""Library to download hitorical trading data."""
from absl import logging
import datetime
import os
import pandas as pd
import requests
import urllib

from typing import Dict

# Create your own API Key in WorldTradingData.com and enter it in this file, under the path.
WTD_API_KEY_FILE_NAME = 'WTD_API_KEY.txt'

_PARSE_ISO_8601_FORMAT = '%Y-%m-%d %H:%M:%S.%f'


class DataManager:
  """Manages download and cache of historical ticker data, using WorldTradingData.com"""

  def __init__(self, base_path: str):
    """Constructor, takes a WorldTradingData key and a base path where to store cached data."""
    self._base = base_path
    self._data: Dict[str, pd.DataFrame] = dict()
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

  def PathForRawData(self, symbol: str) -> str:
    """Creates returns directory for symbol."""
    p = '{}/WTD/{}'.format(self.base, symbol)
    if not os.path.exists(p):
      os.makedirs(p)
    return p

  def _DownloadRawDataToFile(self, symbol: str) -> None:
    """Downloads data for the given symbol given and saves in the local directory."""
    logging.info('Downloading historical data for %s', symbol)
    url = 'https://www.worldtradingdata.com/api/v1/history?symbol={}&output=csv&sort=oldest&api_token={}'.format(
        symbol, self._wtd_key)
    response = requests.get(url)
    data = response.text
    p = self.PathForRawData(symbol)
    with open(p + '/data.csv', 'w') as f:
      f.write(data)
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
    self._data[symbol] = pd.read_csv(p + '/data.csv')

  def SaveData(self, symbol: str) -> None:
    """Saves updated data for given symbol: this is useful to cache also dervided columns. Timestamp is unchanged."""
    if not symbol in self._data:
      raise ValueError(f'SaveData for unknown symbol "{symbol}"')
    df = self._data[symbol]
    p = self.PathForRawData(symbol)
    df.to_csv(p + '/data.csv')
