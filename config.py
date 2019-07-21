# coding=utf-8
# pylint: disable=line-too-long
"""Configuration for the ETF time series, including tickers to include by default."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


TICKERS = [
    'BIV', 'BLV', 'BND', 'BNDW', 'BNDX',
    'BSV', 'EDV', 'IVOG', 'IVOO',
    'IVOV', 'MGC', 'MGK', 'MGV', 'VAW',
    'VB', 'VBK', 'VBR', 'VCIT', 'VCLT',
    'VCR', 'VCSH', 'VDC', 'VDE', 'VEA',
    'VEU', 'VFH', 'VGIT',
    'VGK', 'VGLT', 'VGSH', 'VGT',
    'VHT', 'VIG', 'VIGI', 'VIOG',
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
    'VPU',
    'VSS',
    'VT',
    'VTC', 'VTEB', 'VTHR', 'VTI',
    'VTIP', 'VTV', 'VTWG', 'VTWO', 'VTWV',
    'VUG', 'VV', 'VWO',
    'VWOB', 'VXF', 'VXUS', 'VYM', 'VYMI',

    #

    # European companies.
    'VNRT.L',

    # S&P 500 UCITS ETF (USD) Tracks the performance of the S&P 500.
    'VUSA.L',

    # Strategic metals / materials:
    'REMX',  # The VanEck Vectors Rare Earth/Strategic Metals
    'PALL',  # The ETFS Physical Palladium Shares
    'PICK',  # The iShares MSCI Global Select Metals & Mining Producers ETF
    'XME',   # SPDR S&P Metals and Mining
    'GLD',   # The SPDR Gold Trust (GLD)

]

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

# Symbols not available in WorldTradeData:
# 'ESGV', 'VLFQ', 'VFMF', 'VSGX', 'VFQY', 'VFMO',
# 'VFMV', 'VFVA'
