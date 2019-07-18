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
]


# Symbols not available in WorldTradeData:
# 'ESGV', 'VLFQ', 'VFMF', 'VSGX', 'VFQY', 'VFMO',
# 'VFMV', 'VFVA'
