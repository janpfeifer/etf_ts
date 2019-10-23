# coding=utf-8
# pylint: disable=line-too-long
"""Configuration for the ETF time series, including tickers to include by default."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Sources:
#
# finance.google.com
# finance.yahoo.com
# etfdb.com
# vanguard.com
#
# List of exchange: https://www.google.com/googlefinance/disclaimer/
TICKERS = [

    #================================================================================================
    # Indices
    'BNDW',    # Vanguard Total World Bond ETF, tracks the performance of the Bloomberg Barclays Global Aggregate Float Adjusted Composite Index
    'VNRT.L',  # FTSE North America UCITS ETF (USD) Distributing (VNRT)
    'VOOG',  # Vanguard ADMIRA/S&P 500 GR IX FD ET
    'VOOV',  # Vanguard S&P 500 Value ETF
    'VUSA.L',  # S&P 500 UCITS ETF (USD) Tracks the performance of the S&P 500.

    #================================================================================================
    # Vanguard Retirement Funds
    'VFORX',

    #================================================================================================
    # Swiss ETFs and large companies.
    'CHSPI.SW',  # iShares Core SPI

    # Aka SWX:CSSMIM, iShares SMIM® ETF (CH), The Fund seeks to track
    'CSSMIM.SW',
    # the performance of an index composed of the 30 largest Swiss companies
    # not included in the SMI index

    # iShares SLI, track the performance of an index composed of the 30
    # largest Swiss companies.
    'CSSLI.SW',
    'CSSMI.SW',
    'CHDVD.SW',

    # iShares MSCI Switzerland ETF (in USD): The MSCI Switzerland index tracks
    # around 40 leading stocks on the Swiss market. (in USD)
    'EWL',

    # Solactive Swiss Large Cap
    'XSMC.SW',

    # PSP Swiss Property AG: PSP Swiss Property owns a real estate portfolio with office and commercial
    # properties totalling CHF 7.7 billion.
    # Market Capital 5.7B
    'PSPN.SW',

    # The Swiss Re Group is one of the world's leading providers of reinsurance,
    # insurance and other forms of insurance-based risk transfer, working to make the world more resilient.
    # Market Capital: ~32B
    'SREN.SW',

    #================================================================================================
    # Mixes
    'GSLC',  # ActiveBeta U.S. Large Cap Equity ETF
    'VOT',  # Vanguard Mid-Cap Growth ETF (VOT)
    'VOE',  # VANGUARD IX FUN/VANGUARD MID-CAP VA
    'AIEQ',  # AI Powered Equity ETF, https://etfmg.com/funds/aieq/
    'XLF',  # Financial Select Sector SPDR Fund

    #================================================================================================
    # Strategic metals, materials and water.
    'REMX',  # The VanEck Vectors Rare Earth/Strategic Metals
    'PALL',  # The ETFS Physical Palladium Shares
    'PHPD.L',  # Palladium ETFS PHPD INAV
    'JJN',    # Nickel based ETF.
    'PICK',  # The iShares MSCI Global Select Metals & Mining Producers ETF
    'XME',   # SPDR S&P Metals and Mining, dividend based.
    # 'GLD',   # The SPDR Gold Trust (GLD) -> Equivalent to SGLD.L
    # 'SGOL',  # Aberdeen Standard Investments Physical Swiss Gold Shares421.5440 ETF -> SGLD.L equivalent
    #'ZGLD',  # ZKB Gold ETF AA CHF Klasse (SWX)
    #'ZPAL',  # ZKB ZKB Palladium ETF (SWX)
    'SGLD.L',  # SRC PH/ASST BKD 21001231 SER -> SGOL equivalent
    # Also available in London, but data available only for the Amsterdam.
    'GLDU.AS',

    #================================================================================================
    # Bonds
    'BLV',  # Vanguard Long-Term Bond ETF.
    'EDV',  # Vanguard Extended Duration Treasury ETF -> US extended bonds.
    'IEF',   # iShares 7-10 Year Treasury Bond ETF
    'TLT',   # iShares 20+ Year Treasury Bond ETF
    'GOVT',  # iShares US Treasury Bond ETF
    'VGIT',  # Vanguard Intermiate Term Treasury ETF
    'VGLT',  # Vanguard Long-Term Treasury ETF
    'FEMB',  # First Trust Emerging Markets Local Currency Bond ETF

    #================================================================================================
    # Dividends
    'HDV',   # iShares Core DHigh Dividend HDV
    'DGRO',  # iShares Core Dividend Growth ETF

    #================================================================================================
    # Emerging Markets
    'IEMG',  # iShares Core MSCI Emerging Markets ETF

    #================================================================================================
    # Information Technology
    'PSJ',   # Invesco Dynamic Software ETF
    'IGV',   # iShares Expanded Tech-Software ETF
    'HACK',  # ETF MANAGERS TR/ETFMG PRIME CYBER Security
    'CLOU',  # Global X Cloud Computing ETF
    'XSW',   # SPDR S&P Software & Services ETF
    'GAMR',  # ETFMG Video Game Tech ETF
    'VOX',  # Vanguard Telecommunication Services ETF

    #================================================================================================
    # Health
    'XLV',   # Health Care Select Sector SPDR Fund
    'VHT',   # Vanguard Healthcare ETF
    'IBB',   # iShares Nasdaq Biotechnology ETF
    'XBI',   # SPDR S&P Biotech ETF
    'IHI',   # iShares U.S. Medical Devices ETF

    #================================================================================================
    # Utilities
    'VPU',  # Vanguard Utilities Index Fund ETF Shares
    # 'AWTAX',  # AllianzGI Global Water Fund A (AWTAX)

    #================================================================================================
    # Not yet classified.
    'BIV', 'BND', 'BNDW', 'BNDX',
    'BSV', 'IVOG', 'IVOO',
    'IVOV', 'MGC', 'MGK', 'MGV', 'VAW',
    'VB', 'VBK', 'VBR', 'VCIT', 'VCLT',
    'VCR', 'VCSH', 'VDC', 'VDE', 'VEA',
    'VEU', 'VFH',
    'VGK', 'VGSH', 'VGT',
    'VIG', 'VIGI', 'VIOG',
    'VIOO', 'VIOV', 'VIS', 'VMBS', 'VNQ',
    'VNQI',
    'VO',
    'VONE',
    'VONG',
    'VONV',
    'VOO',
    'VPL',
    'VSS',
    'VT',
    'VTC', 'VTEB', 'VTHR', 'VTI',
    'VTIP', 'VTV', 'VTWG', 'VTWO', 'VTWV',
    'VUG', 'VV', 'VWO',
    'VWOB', 'VXF', 'VXUS', 'VYM', 'VYMI',
]

# Tickers whose hystorical data are to be read from Yahoo Finance.
TICKERS_FROM_YAHOO_FINANCE = set([
    'CSSMI.SW',
    'CSSMIM.SW',
    'CHSPI.SW',
    'CHDVD.SW',
    'CSSLI.SW',
    'XSMC.SW',
    'VFORX',
])

# Default starting of historical data.
DEFAULT_MIN_DATE = '2006-01-01'

# Trim data for some assets that have broken data. For each symbol defines the
# minimum start date to consider.
FIX_MIN_DATE = {
    '100GBA.SW': '2008-01-03',

    '3BRL.L': '2016-06-08',
    'CACX.L': '2014-09-01',
    'CASH.L': '2018-05-16',
    'FXRU.L': '2019-01-17',
    'JPSR.SW': '2018-07-02',
    'LCAN.L': '2018-04-09',
    'LCWL.L': '2019-09-17',
    'MEUG.L': '2018-05-16',
    'MTXX.L': '2017-11-27',
    'VIG': '2006-01-01',
    'VOT': '2006-08-25',
    'WLDD.L': '2012-11-27',

    '100GBA.SW': '2008-01-03',
    '2AAP.L': '2019-01-03',
    '3BLR.L': '2016-08-05',
    '3BSR.L': '2016-07-29',
    '3HCL.L': '2013-08-29',
    '3LGS.L': '2018-11-19',
    '3LNG.L': '2018-11-15',
    '3NGS.L': '2018-11-14',
    '3UKL.L': '2016-06-24',
    'CBAUTO.SW': '2010-10-25',
    'CBHECA.SW': '2014-07-14',
    'CBSX5E.SW': '2014-03-17',
    'CBTRAV.SW': '2014-07-22',
    'CEGY.SW': '2015-02-13',
    'CHGB.L': '2019-09-16',
    'CMU.L': '2018-02-21',
    'CNAL.L': '2018-05-17',
    'CRM2.L': '2018-07-16',
    'E50EUA.SW': '2007-05-15',
    'EEEA.SW': '2018-08-24',
    'EMUEUA.SW': '2008-08-20',
    'ETBB.SW': '2018-07-10',
    'ETZ.SW': '2018-06-13',
    'GENDEG.SW': '2018-09-05',
    'GS2.L': '2019-07-15',
    'IASP.SW': '2008-11-13',
    'IBTM.SW': '2008-01-16',
    'IGIL.SW': '2008-10-24',
    'IGLO.SW': '2009-05-26',
    'ISWD.SW': '2008-12-22',
    'LCAU.L': '2018-04-10',
    'LCOP.L': '2008-10-22',
    'LPET.L': '2008-10-30',
    'LYLVE.SW': '2013-10-11',
    'LYMUA.SW': '2010-05-18',
    'LYMWO.SW': '2010-05-17',
    'SG28.L': '2018-11-28',
    'SG29.L': '2019-04-12',
    'SG43.L': '2019-06-25',
    'SP5G.L': '2019-04-10',
    'USAUSA.SW': '2010-06-03',
    'VIOV': '2015-08-24',
    'VIXL.L': '2016-07-26',
    'VYM': '2006-11-16',
    'WRDUSA.SW': '2010-06-08',
    'ZSILUS.SW': '2014-10-07',

    'LYTSX.SW': '2018-04-05',
    '3NGS.L': '2018-05-16',
    'PSPN.SW': '2001-09-03',
    '3NGL.L': '2017-03-15',
    'CNAL.L': '2018-05-17',
    'EEEA.SW': '2018-08-24',
    'GENDEG.SW': '2018-09-05',
    'IBTM.SW': '2008-01-04',
    'IEMI.SW': '2008-10-16',
    'IGIL.SW': '2008-08-07',
    'IGLO.SW': '2009-03-13',
    'ISWD.SW': '2008-11-12',
    'LCAU.L': '2018-04-10',
    'JJN': '2018-04-24',
    'LAUU.L': '2018-03-26',
    'LAUS.L': '2018-03-22',
}


FIX_MISSING_OPEN = {
    'AWTAX',
}

# Only consider information after this date: before this most ETFs didn't exist.
START_DATE = '2004-01-01'


# Ordered list of fields we are importing to TensorFlow for optimization.
FIELDS_FOR_TENSORFLOW = [
    'Open',
    'Close',
    'High',
    'Low',
    'Volume',
    'MidOpenClose',
    'MidHighLow',
    # 'DiffHighLow',
    'DiffHighLowPct',
    # 'DailyGain',
    'PctDailyGain',
    'LogDailyGain',
    'DeltaSerial',
    'ClosePctVol',
    'PctDailyGainVol',
    'VolumeMean',
]


# A month is roughly equivalent to 17 "days"
MONTHLY_PERIOD_IN_SERIAL = 17

# A year is roughly wequivalent to 240 "days" (serial counts)
YEARLY_PERIOD_IN_SERIAL = 257

# Spread dividend award across the previous days.
SPREAD_DIVIDENDS = 1 * YEARLY_PERIOD_IN_SERIAL

# Symbols not available in WorldTradeData:
# 'ESGV', 'VLFQ', 'VFMF', 'VSGX', 'VFQY', 'VFMO',
# 'VFMV', 'VFVA'

# Period we report.
REPORT_PERIOD_YEARS = 6
REPORT_PERIOD = REPORT_PERIOD_YEARS * YEARLY_PERIOD_IN_SERIAL

# Mix strategy constants:
TRAINING_PERIOD = 4 * YEARLY_PERIOD_IN_SERIAL
APPLYING_PERIOD = YEARLY_PERIOD_IN_SERIAL // 4  # Quarlerly
