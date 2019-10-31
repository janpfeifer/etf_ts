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

    # The following won't be available for Swiss Residents starting on 1/1/2020, according to
    # https://thepoorswiss.com/swiss-investors-lose-access-us-domiciled-etfs/ 
    #
    # See exclude_us_etfs() below.
    {
        # NYSE Arca
        'suffix': None,
        'url': 'https://www.interactivebrokers.com/en/index.php?f=567&exch=arca',
    },
    {
        # NYSE American (AMEX)
        'suffix': None,
        'url': 'https://www.interactivebrokers.com/en/index.php?f=567&exch=amex',
    },
    {
        # NASDAQ OMX BX (BEX)
        'suffix': None,
        'url': 'https://www.interactivebrokers.com/en/index.php?f=567&exch=bex'
    },
]


CURRENCIES = ['GBP', 'CHF', 'USD', 'EUR', 'JPY']

SKIP_SYMBOLS = ([
    # Symbols with some crazy fluctuations (values dropping to 1% of it value for a couple of
    # days?!)
    # '3BRS.L', '3DEL.L', '3DES.L', '3ELM.L', '3EML.L', '3EMS.L', '3EUS.L', '3LDE.L', '3USL.L',
    # '3USS.L', 'AASG.L', 'ACWD.L', 'AIGG.L', 'AUGA.L', 'AUSAUW.SW', 'BATT.L', 'BIOT.L', 'BRIC.SW',
    # 'CAHGBA.SW', 'CARB.L', 'CASH.L', 'CBCOMM.SW', 'CBNKY.SW', 'CBSEU.SW', 'CBU7.L', 'CBUS.SW', 'CBUSIH.SW',
    # 'CCAU.L', 'CE9U.L', 'CEMA.L', 'CES1.L', 'CEU1.L', 'CEUG.L', 'CHIP.L', 'CIND.L', 'CJ1U.L',
    # 'CJPU.L', 'CLIM.L', 'CMXC.L', 'CNAL.L', 'CPXJ.L', 'CRB.SW', 'CRPS.L', 'CSBGU7.SW', 'CSEMAS.SW',
    # 'CSEMU.SW', 'CSH2.L', 'CSINDU.SW', 'CSMIB.SW', 'CSNKY.SW', 'CSP1.L', 'CSPXJ.SW', 'CSSX5E.SW', 'CSUK.SW',
    # 'CSUKX.SW', 'CSUS.SW', 'CSUSS.SW', 'CWEG.L', 'CXAP.L', 'DEMP.L', 'DESD.L', 'DFEE.L', 'DFEP.L',
    # 'DGRG.L', 'DGRP.L', 'DGSE.SW', 'DHSD.L', 'DISG.L', 'DL2P.L', 'DLTM.L', 'DS2P.L', 'DXGE.SW',
    # 'DXGZ.L', 'ECOM.L', 'EEEG.SW', 'EEIE.L', 'EFIE.SW', 'EFIS.L', 'ELLE.L', 'EMCB.L', 'EMIM.L',
    # 'EMIN.L', 'EPVL.L', 'EQDS.L', 'EQLT.L', 'EQQU.L', 'ERN1.L', 'ERNU.L', 'ERO.L', 'ES15.L',
    # 'ESD.SW', 'ESDD.SW', 'ESE.SW', 'ESEH.SW', 'ESM.L', 'ETDD.SW', 'ETSY.L', 'ETZD.SW', 'EUGBA.SW',
    # 'EUGBPD.SW', 'EUSC.L', 'FEMQ.L', 'FEMU.L', 'FEUZ.L', 'FEX.L', 'FEXU.L', 'FGQI.L', 'FING.L',
    # 'FINW.L', 'FLO5.L', 'FTAL.SW', 'FUSA.L', 'GBDV.L', 'GGRG.L', 'GLAB.SW', 'GLCB.L', 'GS1EUA.SW',
    # 'HEDP.SW', 'HEWA.L', 'HEWD.L', 'HMAD.L', 'HMCD.L', 'HMEM.L', 'HMFD.L', 'HMJD.L', 'HMJP.SW',
    # 'HMUD.L', 'HMYD.L', 'HRUD.L', 'HTWD.L', 'HWWA.L', 'IBGX.L', 'ICOV.SW', 'IDEE.L', 'IDPE.L',
    # 'IDTK.L', 'IE15.L', 'IEFM.L', 'IEFS.L', 'IEFV.L', 'IEGZ.SW', 'IEMA.L', 'IESE.SW', 'IESU.L',
    # 'IEXF.SW', 'IGHY.L', 'IGLN.L', 'IGSD.L', 'IGSU.L', 'IHCU.L', 'IHYG.SW', 'IJPA.L', 'IJPN.SW',
    # 'IMEA.SW', 'IMV.L', 'INAA.SW', 'INFG.L', 'IPXJ.L', 'IRCP.SW', 'ISAC.L', 'ISDU.L', 'ISDW.L',
    # 'ISFE.SW', 'ISFR.L', 'ISP6.L', 'ISRL.L', 'IUKD.SW', 'IWFS.L', 'IWFV.L', 'J13U.L', 'JLES.L',
    # 'JPFM.L', 'JPJP.L', 'JPNC.SW', 'JPNJPA.SW', 'JPUSBH.SW', 'JPXG.L', 'JPYP.L', 'L100.L', 'LCJD.L',
    # 'LCNY.L', 'LCUD.L', 'LCUK.L', 'LGCF.L', 'LGCU.L', 'LGOV.SW', 'LOWE.L', 'LUMV.L', 'LUXG.L',
    # 'LWMV.L', 'LYLEM.SW', 'LYMTI.SW', 'LYMTX.SW', 'MATG.L', 'MGTU.L', 'MIBX.L', 'MIDD.SW', 'MINV.L',
    # 'MLPQ.L', 'MLPX.L', 'MMS.SW', 'MSEG.SW', 'MVAU.L', 'MVMX.L', 'MXFP.L', 'NRGG.L', 'OILW.L',
    # 'PHPP.L', 'PQVG.L', 'PSES.L', 'PSRE.L', 'PSRM.L', 'PSWU.L', 'R2SC.L', 'RISE.L', 'ROBG.L',
    # 'RTWO.L', 'S2USBH.SW', 'SBEG.L', 'SBUY.L', 'SEMH.L', 'SEUP.L', 'SF1CHA.SW', 'SG21.L', 'SGLD.L',
    # 'SGLO.L', 'SGQD.L', 'SGQE.L', 'SGVB.L', 'SHYU.L', 'SMMCHA.SW', 'SPLT.L', 'SPPP.L', 'STAG.L',
    # 'SUES.L', 'SUSS.L', 'SW2CHA.SW', 'TELG.L', 'TIP1D.SW', 'TIPU.L', 'TNOG.L', 'TP05.L', 'UB00.L',
    # 'UB01.L', 'UB03.L', 'UB20.L', 'UB30.L', 'UB45.L', 'UBTL.L', 'UBXX.L', 'UC04.L', 'UC07.L',
    # 'UC13.L', 'UC14.L', 'UC44.L', 'UC55.L', 'UC65.L', 'UC68.L', 'UC81.L', 'UC82.L', 'UC86.L',
    # 'UC87.L', 'UC90.L', 'UC93.L', 'UC94.L', 'UC95.L', 'UC96.L', 'UC98.L', 'UC99.L', 'UCAP.L',
    # 'UD03.L', 'UD05.L', 'UDVD.L', 'UGAS.L', 'UIFS.L', 'UKDV.SW', 'UKRE.L', 'UQLT.L', 'US13.L',
    # 'US35.L', 'US71.L', 'USFM.L', 'USMV.L', 'USPY.L', 'USTY.L', 'UTIG.L', 'VDMO.L', 'VDTY.L',
    # 'VDUC.L', 'VECP.SW', 'VEMT.L', 'VEUD.L', 'VILX.L', 'VIXL.L', 'VJPN.SW', 'VMID.SW', 'VNRT.L',
    # 'VUSA.L', 'VWRD.L', 'WCOB.L', 'WOOD.SW', 'WSML.L', 'WSRUSA.SW', 'WTID.L', 'XAD1.SW', 'XAUS.L',
    # 'XBGG.SW', 'XD3E.L', 'XDBG.SW', 'XDER.L', 'XDNS.L', 'XDWG.SW', 'XGDD.L', 'XGID.L', 'XGIG.L',
    # 'XGSG.L', 'XKS2.L', 'XLFS.SW', 'XLIS.SW', 'XLKQ.L', 'XLKS.SW', 'XLPE.L', 'XLPP.L', 'XMAD.L',
    # 'XMCX.L', 'XMMD.L', 'XMTD.L', 'XMUD.L', 'XNID.L', 'XPHG.L', 'XS3R.L', 'XS6R.L', 'XS7R.L',
    # 'XSD2.L', 'XSNR.L', 'XSPS.L', 'XSTR.L', 'XSX6.L', 'XVTD.L', 'XX2D.L', 'ZIEU.L', 'ZWEE.L',
    # 'ZWUU.L',
])

SYMBOL_TO_INFO = {}


def exclude_us_etfs() -> None:
    """Call this before extract_ib_symbols, to exclude US based ETFs"""
    global INTERACTIVE_BROKERS_SOURCES
    INTERACTIVE_BROKERS_SOURCES = INTERACTIVE_BROKERS_SOURCES[:3]


def extract_ib_symbols(base_dir: Text, max_age_days: int= 30) -> List[Text]:
    # Get list of ETFs from IB published lists.
    symbols = set()
    # Prevent same symbol being used from different exchanges.
    raw_symbols = set()

    for source in INTERACTIVE_BROKERS_SOURCES:
        df = _extract_symbols_from_source(
            source['url'], base_dir, max_age_days)
        df_symbols, df_ib_symbols, df_descriptions, df_currencies = (
            df['Symbol'], df['IB Symbol'],
            df['Fund Description (Click link for more details)'],
            df['Currency'])
        for ii in df.index:
            symbol = df_symbols[ii]
            ib_symbol = df_ib_symbols[ii]
            currency = df_currencies[ii]
            description = df_descriptions[ii] + ' (' + currency + ')'
            if symbol[-3:] in CURRENCIES:
                symbol = symbol[: -3]
            if symbol in raw_symbols:
                continue
            if source['suffix']:
                symbol_ex = symbol + '.' + source['suffix']
            else:
                symbol_ex = symbol
            if symbol_ex in SKIP_SYMBOLS:
                continue
            symbols.add(symbol_ex)
            raw_symbols.add(symbol)
            SYMBOL_TO_INFO[symbol_ex] = {
                'ib_symbol': ib_symbol,
                'description': description,
                'currency': currency}

    # Arbitrary selection of symbols: both ETF and stocks.
    for symbol in config.TICKERS:
        if symbol not in symbols:
            symbols.add(symbol)
            SYMBOL_TO_INFO[symbol] = {
                'ib_symbol': '?',
                'description': 'Manually selected.',
                'currency': '?'
            }

    return sorted(list(symbols))


def _extract_symbols_from_source(url: Text,
                                 base_dir: Text,
                                 max_age_days: int= 30):
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
