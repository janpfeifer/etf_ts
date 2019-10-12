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
from typing import Text
import urllib

# Try parsing HTML with pandas:
# tables = pd.read_html("https://www.interactivebrokers.com/en/index.php?f=567&exch=ebs")

# ETFs at SIX Swiss Exchange (EBS)
# https://www.interactivebrokers.com/en/index.php?f=567&exch=ebs
TICKERS_ETFS_AT_SIX = [
]


def extract_ib_symbols(base_dir: Text,
                       url: Text = 'https://www.interactivebrokers.com/en/index.php?f=567&exch=ebs',
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
    print(table)
    return ['a', 'b']
