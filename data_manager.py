"""Library to download hitorical trading data."""
from absl import logging
import datetime
import os
import numpy as np
import pandas as pd
import requests
from typing import Dict, List, Optional, Text, Union
import urllib
from yahoofinancials import YahooFinancials

import config
import config_ib

# Create your own API Key in WorldTradingData.com and enter it in this file, under the path.
WTD_API_KEY_FILE_NAME = 'WTD_API_KEY.txt'

WTD_SYMBOLS_FILES = [
    # Can be downloaded with an upgraded account in https://www.worldtradingdata.com/
    'worldtradingdata-stocklist.csv',
    'worldtradingdata-usmutualfundlist.csv',
]

_PARSE_ISO_8601_FORMAT = '%Y-%m-%d %H:%M:%S.%f'
MIN_NUM_PRICE_VALUES = config.YEARLY_PERIOD_IN_SERIAL / 2


SKIP_SYMBOLS = set([
    # Recent drastic changes in prices.
    'LCWL.L', '3GS.L', '2NVD.L', '2VIS.L', '3BTL.L', 'AEXK.L',
    '5ESGG.SW', 'ACWL.L', 'AT1D.SW', 'BBDD.L', 'BBTR.SW', 'BCHS.L', 'BENE.L', 'BYBG.L', 'C8M.L',
    'CT5.L', 'CU5.L', 'ESM.L', 'FING.L', 'FKUD.L', 'FSKY.L', 'FUSP.SW', 'HEDG.L', 'INTL.L',
    'INUG.L', 'J15R.L', 'JRBE.L', 'JRBU.L', 'KBA.L', 'KWEB.L', 'LCRW.L', 'META.L', 'NRGG.L',
    'PIMT.L', 'PRIC.L', 'PRIG.L', 'PRIJ.L', 'PRIR.L', 'PRIT.L', 'PRIW.L', 'PRIZ.L', 'SG20.L',
    'SGQG.L', 'SGQX.L', 'SJPP.L', 'STPU.L', 'VEUA.L', 'WELL.L', 'LYAU2.SW', 'LAUS.L',

    # Insufficient / stale data.
    # 'BRZ.SW',

    # Removed due to insufficient data: could revisit later.
    '2GS.L', 'BBLL.L', 'BCHN.L', 'CBGB10.SW', 'CBTPX.SW', 'CE71.L', 'CI2G.L', 'CS5.L', 'CUIH.L',
    'DGRG.L', 'DOCT.L', 'DXGZ.L', 'EEUD.L', 'ETSY.L', 'FJP.L', 'FRCH.L', 'FRIN.L', 'J13E.L',
    'JMFP.SW', 'KLWD.L', 'KRWL.L', 'LMMV.L', 'LWMV.L', 'OGSC.L', 'S2HGBD.SW', 'SMRG.L',
    'TELG.L', 'TR3G.L', 'TR7S.SW', 'TRXG.L', 'UBIF.L', 'UC48.L', 'VDCA.L', 'VMIG.L',
    'VNRA.L', 'VUAA.L', 'VUAG.L', 'VUKG.L', 'WCLD.L', 'WCOM.SW', 'XAGG.L',
    'GLDU.SW'
])

YAHOO_SKIP_SYMBOLS = set([
    # Missing from Yahoo
    'CE8.SW', 'BBIL.SW', 'CI2.SW', 'TCMCI.SW', 'T3GB.SW', 'TAGCI.SW', 'JPGL.SW', 'CC1.SW',
    'WGOV.SW', '500.SW', 'CNYB.SW', 'EYLD.SW', 'CWE.SW', 'I.SW', 'CW8.SW', 'GOVE.SW', 'TRRCI.SW',
    'ESGGB.SW', 'CHSRIA.SW', 'CE9.SW', 'CWF.SW', 'CBM.SW', 'TCNCI.SW', 'TWWCI.SW', 'USDE.SW',
    'TRGB.SW', 'A.SW', 'TLPCI.SW', 'WYLD.SW', 'OIL2S.SW', 'EMHE.SW', 'DBXI.SW', 'CU2.SW', 'CSW.SW',

    'HPRD.SW', 'XAIX.SW', 'UCAP.SW', 'WCOG.SW', 'VLUD.SW', 'LYRUD.SW', 'RQFI.SW', 'XG7S.SW',
    'HODL.SW', 'XIEE.SW', 'XGII.SW', 'CAPU.SW', 'SP500F.SW', 'SHEMG.SW', 'SPXD.SW', 'ICHN.SW',
    'EMLOC.SW', 'SWRD.SW', 'CBUSAC.SW',

    '0MTJ.L', '020Y.L', '0FLE.L', '0GGH.L', 'GLAD.L', '0H1I.L', 'OWLP.L', 'A.L', '0W6Q.L', 'PRCU.L',
    'LOUF.L', '0MT6.L', 'IQSA.L', 'T3GB.L', 'SYB3.L', 'FEME.L', 'FEUI.L', '0MUL.L', 'IEMU.L', 'S.L',
    '0MTG.L', 'F.L',

    '0Y7Z.L', '0W7A.L', 'ULV2.L', '0E0Y.L', '0HC6.L', '0W7N.L', '0Y22.L', '0H71.L', '0H80.L',
    '0JRY.L', '0MUX.L', '0W7I.L', '0H82.L', '0Y49.L', 'CLMU.L', '0HCD.L', '0H7Q.L', '0ANL.L',
    '0H6R.L', '0HCS.L', '0H7L.L', '0Y7Q.L', '0ANU.L', '0E1M.L', '0HDN.L', '0JGK.L', '0MUR.L',
    '0GP0.L', '0W7K.L', '0Y7H.L', '0H8Y.L', '0L12.L', '0LOE.L', '0MVS.L', 'OAUH.L', '0IAS.L',
    '0I9M.L', '0H9C.L', '0Y8Z.L', '0H6V.L', '0W7X.L', '0DKR.L', 'L.L', '0H70.L', '0KQK.L', '0L13.L',
    '0YEB.L', '0HBV.L', '0Y6W.L', '0XE0.L', '0Y0J.L', '0AO7.L', 'GOOE.L', '0H6Z.L', 'MSRG.L',
    '0KR4.L', '0H9I.L', '0HAX.L', 'GS2E.L', 'FIND.L', '0Y7I.L', 'RDS2.L', '0Y8I.L', '0H97.L',
    '0H7Y.L', '0LNF.L', '0Y2M.L', '0W3K.L', 'CITE.L', '0LLL.L', 'CRME.L', '0H78.L', '0ANK.L',
    '0MOL.L', 'V.L', '0H77.L', '0IAW.L', 'ESRG.L', '0HFU.L', '0MWL.L', '0HCP.L', '0MJI.L',
    '0RSJ.L', 'SBD1.L', 'JP2E.L', '0H6U.L', '0H8Z.L', '0H6S.L', '0HFX.L', '0YWV.L', '0HD7.L',
    '0H72.L', 'O8PS.L', '0ANW.L', 'TREG.L', '0Y1X.L', '0ANP.L', 'TNAE.L', '0HB3.L', '0H6P.L',
    '0Y2D.L', '0W76.L', '0HDC.L', '0Y4G.L', '0VR9.L', '0ANI.L', '0HDW.L', '0H6Q.L', '0HEX.L',
    '0H8Q.L', '0Y37.L', '0W7Z.L', '0Y7R.L', '0Y7J.L', '0HA2.L', '0HD1.L', '0LO7.L', 'AZN2.L',
    'OSPI.L', '0I9S.L', '0HAP.L', '0Y4N.L', '0HFZ.L', '0H7T.L', '0HDM.L', '0HE0.L', 'VISE.L',
    '0HDB.L', '0HES.L', 'X.L', '0E7Q.L', 'VJPA.L', '0H74.L', '0H7N.L', '0HCU.L', 'MIST.L',
    'ENL2.L', '0HA7.L', '0IX4.L', '0FBL.L', '0H8W.L', '0MJQ.L', '0MPZ.L', '0H76.L', '0ANS.L',
    'TVOL.L', '0H7K.L', '0HBG.L', 'TNGB.L', '0Y43.L', '0ANH.L', '0Y7G.L', '0HBL.L', '0HCJ.L',
    '0Y2F.L', '0JH7.L', '0W77.L', '0JKD.L', '0HCF.L', '0JLR.L', 'VOD2.L', '0HCY.L', '0HDZ.L',
    '0GOZ.L', 'MSRU.L', '0HCA.L', '0Y26.L', '0MQ3.L', '0IYH.L', '0YEG.L', '0ANT.L', '0HBW.L',
    '0Y82.L', '0H7W.L', '0YDK.L', '0AND.L', '0Y2B.L', 'ESRU.L', '0HFT.L', '0JLW.L', '0HD4.L',
    '0MWZ.L', '0MJP.L', '0E7L.L', '0HE1.L', 'TDIV.L', '0H7E.L', '0H7P.L', '0H7J.L', '0W28.L',
    '0YE9.L', '0H6L.L', 'TRET.L', 'OGG9.L', '0E7J.L', '0Y3I.L', 'VDPG.L', '0AO0.L', '0MJM.L',
    '0ANY.L', '0W73.L', '0W84.L', 'TCEG.L', '0JJP.L', '0H7M.L', '0Y5Z.L', 'C.L', 'OSB1.L',
    '0HDV.L', '0MPW.L', '0MWX.L', 'BP2.L', 'NFLE.L', '0MV9.L', '0H6W.L', '0YLL.L', '0HCL.L',
    '0H75.L', '0HFW.L', 'OGB5.L', 'CNSG.L', '0W56.L', '0A1I.L', '0H7C.L', '0HFV.L', '0LLR.L',
    'NVDE.L', '0ANO.L', '0ANM.L', 'SBE1.L', '0ANX.L', 'GSK2.L', '0HCG.L', '0MJN.L', '0HAY.L',
    '0YEE.L', '0H7F.L', '0MNZ.L', '0HGS.L', '0MU1.L', 'OAUE.L', '0DT2.L', '0HCW.L', '0JJ4.L',

    '0JLK.L', 'CMGU.L', '0GP2.L', 'OXCT.L', 'TSGB.L', '0VQY.L', 'VFEA.L', 'TDGB.L', 'PRIE.L',
    'CXAD.L', 'XGGB.L', 'OXA1.L', 'VHYA.L', 'AMZE.L', 'XBLC.L', 'LXUU.L', 'LIVU.L', 'OILU.L',
    '0HH0.L', 'SBUS.L', 'VERE.L', 'PRUC.L', 'SILP.L', 'OXA6.L', 'TCBT.L', 'LIVP.L', '0MKV.L',
    '0W0W.L', 'BGX.L', 'OILG.L', '0VOO.L', '0MKU.L', 'OXA2.L', 'NIKU.L', '0L4R.L', 'MCUU.L',
    'BRTG.L', 'BRTU.L', 'CMGG.L', '0MKJ.L', 'VAPU.L', '0Y92.L', 'OVZB.L', 'TEET.L', '0YED.L',
    'CORU.L', 'TCGB.L', '0JFI.L', '0Y8Q.L', 'OGSC.L', 'PLTU.L', 'VHVG.L', '0Y8R.L', 'OVZC.L',
    'MSFE.L', 'TAGB.L', 'AAPE.L', '0LMD.L', 'VHVE.L', '0W81.L', '0L4T.L', '5HEP.L', 'PRMG.L',
    'PRMU.L', 'TGGB.L', '0MPV.L', '0JHS.L', 'VHYG.L', 'BRT3.L', '0IJL.L', '0HOW.L', '0LMN.L',
    'OGY3.L', '0ANQ.L', 'WWTU.L', '0A12.L', 'BBRT.L', '0JFD.L', 'CMBU.L', 'MFEX.L', '0VR7.L',
    'EMXC.L', '0JG6.L', '0IND.L', 'OUES.L', '0MVO.L', 'OGZU.L', '5WTI.L', '0H7U.L', '0JJC.L',
    'BBGE.L', 'SIUS.L', '0VRD.L', '0MS1.L', 'MALX.L', 'LCAL.L', '0VQG.L', 'TEGB.L', 'VAAA.L',
    '0JH3.L', '0H7A.L', 'INMG.L', 'VFEG.L', 'OIL3.L', '0U5Z.L', '0W9D.L', '0VQL.L', '0HET.L',
    'CMCI.L', '0VQK.L', '0H7Z.L', 'CXAS.L', '0MTF.L', 'RIEG.L', '0Y2G.L', '0JKH.L', 'M9SV.L',
    'OEUL.L', '0L0S.L', '0MVW.L', 'OGYL.L', 'RIEU.L', 'PRIU.L', '0MOO.L', 'AGRU.L', '0IYF.L',
    '0W7R.L', 'TGBT.L', 'INMU.L', 'TSWE.L', '0L4Q.L', 'TGBG.L', 'GCGB.L', 'ENRU.L', '0W7P.L',
    'NGAU.L', '0YLF.L', 'ENGB.L', '0KQR.L', 'OFR3.L', '0HBS.L', 'FB2E.L', '0A1T.L', '0JLS.L',
    '0IBR.L', '5HED.L', 'CMBG.L', 'CARP.L', 'AGGB.L', 'NGAG.L', 'CMGB.L', '0VQN.L', 'CMCU.L',
    '0MKF.L', '0JF2.L', 'ALYU.L', '0V08.L', 'COFU.L', 'XHYG.L', '0JLA.L', '0YIK.L', 'OVZA.L',
    'OXA5.L', '0IEQ.L', 'OGLD.L', 'TGET.L', '0HE5.L', '0W7O.L',

    '2JPM.L', '5HEP.L', 'MWJ.L'

    '3LDE.L', 'AUSAUW.SW', 'BRIC.SW', 'CAHGBA.SW', 'CASH.L', 'CBNKY.SW', 'CBSEU.SW', 'CBUS.SW', 'CBUSIH.SW',
    'CRB.SW', 'CSBGU7.SW', 'CSEMAS.SW', 'CSEMU.SW', 'CSINDU.SW', 'CSMIB.SW', 'CSNKY.SW', 'CSPXJ.SW', 'CSSX5E.SW',
    'CSUK.SW', 'CSUKX.SW', 'CSUS.SW', 'CSUSS.SW', 'DGSE.SW', 'DXGE.SW', 'EEEG.SW', 'EFIE.SW', 'ESD.SW',
    'ESDD.SW', 'ESE.SW', 'ESEH.SW', 'ETDD.SW', 'ETZD.SW', 'EUGBA.SW', 'EUGBPD.SW', 'FTAL.SW', 'GLAB.SW',
    'GS1EUA.SW', 'HEDP.SW', 'HMJP.SW', 'ICOV.SW', 'IEGZ.SW', 'IESE.SW', 'IEXF.SW', 'IHYG.SW', 'IJPN.SW',
    'IMEA.SW', 'INAA.SW', 'IRCP.SW', 'ISFE.SW', 'IUKD.SW', 'JPNC.SW', 'JPNJPA.SW', 'LGOV.SW', 'LYLEM.SW',
    'LYMTI.SW', 'LYMTX.SW', 'MIDD.SW', 'MMS.SW', 'MSEG.SW', 'NRGG.L', 'PIMT.L', 'SJPP.L', 'SW2CHA.SW',
    'TIP1D.SW', 'UKDV.SW', 'VECP.SW', 'VFORX', 'VJPN.SW', 'VMID.SW', 'WOOD.SW', 'XAD1.SW', 'XBGG.SW',
    'XDBG.SW', 'XDWG.SW',

    'VFORX',

    'TSWE.SW', 'TGBT.SW', 'TEET.SW', 'TRET.SW', 'TCBT.SW', 'TNAE.SW', 'TDIV.SW', 'VAAA.SW', 'TGET.SW',
    'MKUW.L', 'GLRA.L', 'EGOV.L', 'XZEM.L', 'CHGB.L',

    # Broken
    'SMICHA.SW',
])

WTD_SKIP_SYMBOLS = set([
    '2JPM.L', '5HEP.L', 'MWJ.L', 'VFORX', 'JPSR.SW',
])


class DataManager:
  """Manages download and cache of historical ticker data, using WorldTradingData.com"""

  def __init__(self, base_path: str, load_wtd_symbols: bool = True):
    """Constructor, takes a WorldTradingData key and a base path where to store cached data."""
    self._base = base_path
    self._data: Dict[str, pd.DataFrame] = dict()
    self._total_assets: Dict[str, Optional[float]] = dict()
    self._dividends: Dict[str, pd.DataFrame] = dict()
    with open('{}/{}'.format(base_path, WTD_API_KEY_FILE_NAME), 'r') as file:
      self._wtd_key = file.read().replace('\n', '')

    self._wtd_available_symbols = set()
    if load_wtd_symbols:
      for file_name in WTD_SYMBOLS_FILES:
        logging.info(f'Reading {file_name} with list of assets held in WorldTradingData.')
        p = '{}/{}'.format(base_path, file_name)
        df = pd.read_csv(p)
        for _, row in df.iterrows():
          self._wtd_available_symbols.add(row['Symbol'])
          currency = row['Currency'] if 'Currency' in row else 'USD'
          config_ib.SYMBOL_TO_INFO[row['Symbol']] = {
              'description': f'{row["Name"]} ({currency})',
              'currency': currency,
          }

  @property
  def base(self):
    """Returns base path used for storage."""
    return self._base

  @property
  def data(self) -> Dict[str, pd.DataFrame]:
    """Returns a dictionary of symobl name to Pandas DataFrame with all data downloaded so far."""
    return self._data

  @property
  def dividends(self) -> Dict[str, pd.DataFrame]:
    """Returns a dictionary of symobl name to Pandas DataFrame with dividends information."""
    return self._dividends

  @property
  def total_assets(self) -> Dict[str, Optional[float]]:
    """Returns lastest update on total assets for ETFs (or None for non-ETFs, or ETFs with missing infromation)."""
    return self._total_assets


  def PathForRawData(self, symbol: str) -> str:
    """Creates returns directory for symbol."""
    p = '{}/WTD/{}'.format(self.base, symbol)
    if not os.path.exists(p):
      os.makedirs(p)
    return p

  def _ValidSymbol(self, symbol: Text) -> bool:
    if symbol in SKIP_SYMBOLS or (self._FromYahoo(symbol) and symbol in YAHOO_SKIP_SYMBOLS):
      return False
    return True

  def _FromYahoo(self, symbol: Text) -> bool:
    return (symbol not in self._wtd_available_symbols) or (symbol in WTD_SKIP_SYMBOLS)

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
    csv_data = []
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    all_assets = YahooFinancials(symbols)
    all_dividends = all_assets.get_daily_dividend_data(
        start_date=config.DEFAULT_MIN_DATE, end_date=today)
    for symbol in symbols:
      dividends = all_dividends[symbol]
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

  def _DownloadTotalAssetsFromYahooFinance(self, symbols: List[Text]) -> List[Optional[Text]]:
    csv_data = []
    assets = YahooFinancials(symbols)
    summaries = assets.get_summary_data()
    for symbol in symbols:
      today = datetime.datetime.now().strftime("%Y-%m-%d")
      if symbol in summaries and 'totalAssets' in summaries[symbol] and summaries[symbol]['totalAssets'] is not None: 
        lines = ['Date,Amount']  # Header
        total_assets = summaries[symbol]['totalAssets']
        csv_data.append(f'Date,TotalAssets\n{today},{total_assets}\n')
      else:
        csv_data.append(None)
    return csv_data

  def _LoadData(self, symbol: Text) -> bool:
    """Loads data for symbol, returns `False` if not enough data worth using."""
    # Convert CSV file to a Pandas dataframe.
    p = self.PathForRawData(symbol)
    df = pd.read_csv(p + '/data.csv')
    if 'Date' not in df or len(df['Date']) == 0:
      logging.info(f'Failed to load data for {symbol}')
      return False

    # Filter out rows with missing data.
    for field in ['Close', 'Date']:
      df = df.loc[~df[field].isna()]

    # Reset index so it properly is a range from 0 to len(df)-1
    df = df.reset_index(drop=True)

    # Filter out some odd rows (with very different values, I assume a bug)
    close = df['Close'].values
    close_minus_1 = np.roll(close, -1)
    close_plus_1 = np.roll(close, +1)
    extremes = np.logical_and(_extremes(close, close_minus_1),
                              _extremes(close, close_plus_1))

    if 'Open' in df and df['Open'].shape[0] > 20:
      open_v = df['Open'].values
      open_minus_1 = np.roll(open_v, -1)
      open_plus_1 = np.roll(open_v, +1)
      open_plus_1[0] = open_plus_1[1]
      extremes = np.logical_or(extremes,
                               np.logical_and(_extremes(open_v, open_minus_1),
                                              _extremes(open_v, open_plus_1)))

    if np.any(extremes):
      w = np.where(extremes)[0]
      logging.info(f'Found extreme differences in {symbol}: dropping {len(w)} rows {w}')
      df = df.loc[~extremes]
      # Reset index so it properly is a range from 0 to len(df)-1
      df = df.reset_index(drop=True)

    # Drastic change in values are still likely wrong (there are so much of this
    # broken data ...)
    if 'Open' in df and df['Open'].shape[0] > 20:
      open_v = df['Open'].values
      open_plus_1 = np.roll(open_v, +1)
      open_plus_1[0] = open_plus_1[1]
      open_changes = _extremes(open_v, open_plus_1)
      if np.any(open_changes):
        dates = []
        for idx in np.where(open_changes):
          dates.append(df['Date'][idx].iat[0])
        logging.error(f'There is a skip of prices: \'{symbol}\': \'{dates}\'')
        return False

    if len(df.index) < config.YEARLY_PERIOD_IN_SERIAL:
      return False


    # Load dividends.
    self._data[symbol] = df
    if os.path.isfile(p + '/dividends.csv'):
      dividends = pd.read_csv(p + '/dividends.csv')
    else:
      dividends = None
    self._dividends[symbol] = dividends

    # Load total assets
    fp = p + '/total_assets.csv'
    if os.path.isfile(fp):
      total_assets = float(pd.read_csv(fp)['TotalAssets'][0])
    else:
      total_assets = None
    self._total_assets[symbol] = total_assets

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
    log_symbols('Making data available', symbols)

    # Remove symbols not available.
    available_symbols = []
    ignored_symbols = []
    for symbol in symbols:
      if self._ValidSymbol(symbol):
        available_symbols.append(symbol)
      else:
        ignored_symbols.append(symbol)
    symbols = available_symbols
    log_symbols('After filter, looking', symbols)
    if ignored_symbols:
      log_symbols('  Unavailable', ignored_symbols)

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
      log_symbols('Downloading data from WTD', wtd_symbols)
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
    if need_downloading:
      log_symbols('Downloading dividends from Yahoo', wtd_symbols)
      all_dividends = self._DownloadDividendsCSVFromYahooFinance(need_downloading)
      for (ii, symbol) in enumerate(need_downloading):
        csv = all_dividends[ii]
        if csv is not None:
          p = self.PathForRawData(symbol)
          with open(p + '/dividends.csv', 'w') as f:
            f.write(csv)

      log_symbols('Downloading total assets from Yahoo', wtd_symbols)
      all_assets = self._DownloadTotalAssetsFromYahooFinance(need_downloading)
      for (ii, symbol) in enumerate(need_downloading):
        csv = all_assets[ii]
        if csv is not None:
          p = self.PathForRawData(symbol)
          with open(p + '/total_assets.csv', 'w') as f:
            f.write(csv)

    # Save timestamp.
    if need_downloading:
      log_symbols('Saving timestamp', need_downloading)
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
    log_symbols('Loading data from disk', symbols)
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


def _extremes(val: np.array, val_delta: np.array) -> np.array:
  return np.minimum(val, val_delta) < 0.60 * np.maximum(val, val_delta)
