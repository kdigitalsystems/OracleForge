"""Unit tests for update_tickers.py's leveraged/inverse fund filter."""
from __future__ import annotations

import unittest

from update_tickers import is_leveraged_or_inverse


class LeveragedOrInverseTests(unittest.TestCase):

    def test_vix_futures_products_excluded(self):
        self.assertTrue(is_leveraged_or_inverse('ProShares Ultra VIX Short-Term Futures ETF'))
        self.assertTrue(is_leveraged_or_inverse('iPath Series B S&P 500 VIX Short-Term Futures ETN'))

    def test_ultrashort_and_leveraged_multiplier_excluded(self):
        self.assertTrue(is_leveraged_or_inverse('ProShares UltraShort QQQ'))
        self.assertTrue(is_leveraged_or_inverse('GraniteShares 2x Long MSFT Daily ETF'))
        self.assertTrue(is_leveraged_or_inverse('Direxion Daily Semiconductor Bull 3X Shares'))

    def test_issuer_plus_directional_keyword_excluded(self):
        self.assertTrue(is_leveraged_or_inverse('ProShares Short S&P500'))

    def test_ordinary_equities_and_funds_not_excluded(self):
        self.assertFalse(is_leveraged_or_inverse('NVIDIA Corporation'))
        self.assertFalse(is_leveraged_or_inverse('Apple Inc.'))
        self.assertFalse(is_leveraged_or_inverse('iShares Short Treasury Bond ETF'))
        self.assertFalse(is_leveraged_or_inverse('Invesco Optimum Yield Diversified Commodity Strategy Fund'))
        self.assertFalse(is_leveraged_or_inverse('ARK Innovation ETF'))

    def test_empty_name_not_excluded(self):
        self.assertFalse(is_leveraged_or_inverse(''))
        self.assertFalse(is_leveraged_or_inverse(None))


if __name__ == '__main__':
    unittest.main()
