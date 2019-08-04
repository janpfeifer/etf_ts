# coding=utf-8
# pylint: disable=line-too-long
"""Configuration for the ETF time series, including tickers to include by default."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Sources:
#
# finance.google.com
# etfdb.com
# vanguard.com
#

TICKERS = [
    'BIV', 'BLV', 'BND', 'BNDW', 'BNDX',
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
    'VOE',
    'VONE',
    'VONG',
    'VONV',
    'VOO',
    'VOOG',
    'VOOV',  # Vanguard S&P 500 Value ETF
    'VOT',  # Vanguard Mid-Cap Growth ETF (VOT)
    'VOX',
    'VPL',
    'VSS',
    'VT',
    'VTC', 'VTEB', 'VTHR', 'VTI',
    'VTIP', 'VTV', 'VTWG', 'VTWO', 'VTWV',
    'VUG', 'VV', 'VWO',
    'VWOB', 'VXF', 'VXUS', 'VYM', 'VYMI',

    # Indices
    'VNRT.L',  # FTSE North America UCITS ETF (USD) Distributing (VNRT)
    'VUSA.L',  # S&P 500 UCITS ETF (USD) Tracks the performance of the S&P 500.
    'BNDW',    # Vanguard Total World Bond ETF, tracks the performance of the Bloomberg Barclays Global Aggregate Float Adjusted Composite Index

    # Mixes
    'GSLC',  # ActiveBeta U.S. Large Cap Equity ETF
    'AIEQ',  # AI Powered Equity ETF, https://etfmg.com/funds/aieq/

    # Bonds.
    'EDV',  # Vanguard Extended Duration Treasury ETF -> US extended bonds.

    # Strategic metals, materials and water.
    'REMX',  # The VanEck Vectors Rare Earth/Strategic Metals
    # 'PALL',  # The ETFS Physical Palladium Shares -> Equivalent to PHPD.L
    'PICK',  # The iShares MSCI Global Select Metals & Mining Producers ETF
    'XME',   # SPDR S&P Metals and Mining
    # 'GLD',   # The SPDR Gold Trust (GLD) -> Equivalent to SGLD.L
    # 'SGOL',  # Aberdeen Standard Investments Physical Swiss Gold Shares421.5440 ETF -> SGLD.L equivalent
    #'ZGLD',  # ZKB Gold ETF AA CHF Klasse (SWX)
    #'ZPAL',  # ZKB ZKB Palladium ETF (SWX)
    'PHPD.L',  # Palladium ETFS PHPD INAV -> Equivalent to PALL
    'SGLD.L',  # SRC PH/ASST BKD 21001231 SER -> SGOL equivalent

    # Bonds
    'IEF',   # iShares 7-10 Year Treasury Bond ETF
    'TLT',   # iShares 20+ Year Treasury Bond ETF
    'GOVT',  # iShares US Treasury Bond ETF
    'VGIT',  # Vanguard Intermiate Term Treasury ETF
    'VGLT',  # Vanguard Long-Term Treasury ETF
    'FEMB',  # First Trust Emerging Markets Local Currency Bond ETF
    # 'AWTAX',  # AllianzGI Global Water Fund A (AWTAX)

    # Information Technology
    'PSJ',   # Invesco Dynamic Software ETF
    'IGV',   # iShares Expanded Tech-Software ETF
    'HACK',  # ETF MANAGERS TR/ETFMG PRIME CYBER Security
    'CLOU',  # Global X Cloud Computing ETF
    'XSW',   # SPDR S&P Software & Services ETF
    'GAMR',  # ETFMG Video Game Tech ETF

    # Health
    'XLV',   # Health Care Select Sector SPDR Fund
    'VHT',   # Vanguard Healthcare ETF
    'IBB',   # iShares Nasdaq Biotechnology ETF
    'XBI',   # SPDR S&P Biotech ETF
    'IHI',   # iShares U.S. Medical Devices ETF

    # Utilities
    'VPU',   # Vanguard Utilities Index Fund ETF Shares
]

# Trim data for some assets that have broken data. For each symbo defines the
# minimum start date to consider.
FIX_MIN_DATE = {
    'VIG': '2006-01-01',
    'VOT': '2006-08-25',
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

# Symbols not available in WorldTradeData:
# 'ESGV', 'VLFQ', 'VFMF', 'VSGX', 'VFQY', 'VFMO',
# 'VFMV', 'VFVA'

# Period we report.
REPORT_PERIOD_YEARS = 10
REPORT_PERIOD = REPORT_PERIOD_YEARS * YEARLY_PERIOD_IN_SERIAL

# Mix strategy constants:
TRAINING_PERIOD = 4 * YEARLY_PERIOD_IN_SERIAL
APPLYING_PERIOD = YEARLY_PERIOD_IN_SERIAL // 4  # Quarlerly
