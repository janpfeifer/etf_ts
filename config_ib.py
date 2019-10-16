# coding=utf-8
# pylint: disable=line-too-long
"""List of available tickers for InteractiveBrokers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import hashlib
import pandas as pd
import os
import time
from typing import List, Text
import urllib

import config

# Try parsing HTML with pandas:
# tables = pd.read_html("https://www.interactivebrokers.com/en/index.php?f=567&exch=ebs")

INTERACTIVE_BROKERS_SOURCES = [
    {
        # ETFs at SIX Swiss Exchange (EBS)
        'suffix': 'SW',
        'url': 'https://www.interactivebrokers.com/en/index.php?f=567&exch=ebs',
    },
    {
        # London Stock Exchange (LSE) - ETF Exchange
        'suffix': 'L',
        'url': 'https://www.interactivebrokers.com/en/index.php?f=567&exch=lseetf',
    },
    {
        # London Stock Exchange (LSE)
        'suffix': 'L',
        'url': 'https://www.interactivebrokers.com/en/index.php?f=567&exch=lse',
    },
]


CURRENCIES = ['GBP', 'CHF', 'USD', 'EUR', 'JPY']

SKIP_SYMBOLS = set([
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
])


SYMBOL_TO_INFO = {}


def extract_ib_symbols(base_dir: Text, max_age_days: int = 30) -> List[Text]:
    # Get list of ETFs from IB published lists.
    symbols = set()
    # Prevent same symbol being used from different exchanges.
    raw_symbols = set()

    for source in INTERACTIVE_BROKERS_SOURCES:
        df = _extract_symbols_from_source(
            source['url'], base_dir, max_age_days)
        df_symbols, df_ib_symbols, df_descriptions = (
            df['Symbol'], df['IB Symbol'],
            df['Fund Description (Click link for more details)'])
        for ii in df.index:
            symbol = df_symbols[ii]
            ib_symbol = df_ib_symbols[ii]
            description = df_descriptions[ii]
            if symbol[-3:] in CURRENCIES:
                symbol = symbol[:-3]
            if symbol in raw_symbols:
                continue
            symbol_ex = symbol + '.' + source['suffix']
            if symbol_ex in SKIP_SYMBOLS:
                continue
            symbols.add(symbol_ex)
            raw_symbols.add(symbol)
            SYMBOL_TO_INFO[symbol_ex] = {
                'ib_symbol': ib_symbol,
                'description': description,
            }

    # Arbitrary selection of symbols: both ETF and stocks.
    for symbol in config.TICKERS:
        if symbol not in symbols:
            symbols.add(symbol)
            SYMBOL_TO_INFO[symbol] = {
                'ib_symbol': '?',
                'description': 'Manually selected.',
            }

    return sorted(list(symbols))


def _extract_symbols_from_source(url: Text,
                                 base_dir: Text,
                                 max_age_days: int = 30):
    """Extract list of symbols from URL. Caches page for `max_age_days`."""
    url_hash = hashlib.md5(url.encode()).hexdigest()
    ib_dir = '{}/ib_html_cache'.format(base_dir)
    if not os.path.exists(ib_dir):
        os.makedirs(ib_dir)

    url_file = '{}/{}'.format(ib_dir, url_hash)
    logging.info('URL={}, file={}'.format(url, url_file))

    reuse = False
    max_age_seconds = max_age_days * 24 * 3600
    if not os.path.exists(url_file) or time.time() - os.path.getmtime(url_file) > max_age_seconds:
        logging.info('Fetching data from {}'.format(url))
        response = urllib.request.urlopen(url)
        html = response.read().decode('utf-8')
        with open(url_file, 'w') as f:
            f.write(html)
    else:
        age_seconds = int(time.time() - os.path.getmtime(url_file))
        logging.info('Reading data from file {} (age={}s)'.format(
            url_file, age_seconds))
        with open(url_file, 'r') as f:
            html = f.read()

    tables = pd.read_html(html, match='Symbol.*')
    if len(tables) != 1:
        raise ValueError(
            "There are {} tables matching in {}, contents in {}".format(len(tables), url, url_file))
    table = tables[0]
    return table
