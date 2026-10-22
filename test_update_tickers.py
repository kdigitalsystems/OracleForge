"""Unit tests for update_tickers.py's leveraged/inverse fund filter."""
from __future__ import annotations

import unittest

from unittest.mock import MagicMock

from update_tickers import (
    MIN_UNIVERSE_FRACTION,
    get_tradable_symbols,
    is_leveraged_or_inverse,
    universe_shrunk_too_far,
)


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


class TradableSymbolsTests(unittest.TestCase):

    @staticmethod
    def _asset(symbol, name='Acme Corp', exchange='NYSE', tradable=True, fractionable=True):
        return {'symbol': symbol, 'name': name, 'exchange': exchange, 'tradable': tradable, 'fractionable': fractionable}

    def test_an_exchange_code_the_library_does_not_know_does_not_break_parsing(self):
        # 2026-10-20: Alpaca started returning TXSE; alpaca-py's AssetExchange
        # enum rejects it, which took the nightly down for three days.
        client = MagicMock()
        client.get.return_value = [
            self._asset('AAA'), self._asset('TXS', exchange='TXSE'), self._asset('NONAME', name=None),
        ]
        self.assertEqual(get_tradable_symbols(client), ['AAA', 'TXS', 'NONAME'])
        client.get.assert_called_once_with('/assets', {'status': 'active', 'asset_class': 'us_equity'})

    def test_filters_untradable_nonfractional_and_leveraged_funds(self):
        client = MagicMock()
        client.get.return_value = [
            self._asset('OK'), self._asset('HALT', tradable=False), self._asset('WHOLE', fractionable=False),
            self._asset('SQQQ', name='ProShares UltraPro Short QQQ'),
            self._asset('UVXY', name='ProShares Ultra VIX Short-Term Futures ETF'),
        ]
        self.assertEqual(get_tradable_symbols(client), ['OK'])


class UniverseGuardTests(unittest.TestCase):

    def test_a_large_shrink_is_treated_as_a_data_problem(self):
        self.assertTrue(universe_shrunk_too_far(100, 193))
        self.assertTrue(universe_shrunk_too_far(0, 193))

    def test_normal_drift_and_growth_are_accepted(self):
        self.assertFalse(universe_shrunk_too_far(180, 193))
        self.assertFalse(universe_shrunk_too_far(193, 193))
        self.assertFalse(universe_shrunk_too_far(250, 193))
        self.assertFalse(universe_shrunk_too_far(int(193 * MIN_UNIVERSE_FRACTION) + 1, 193))

    def test_no_previous_universe_means_nothing_to_compare_against(self):
        self.assertFalse(universe_shrunk_too_far(0, 0))
        self.assertFalse(universe_shrunk_too_far(5, 0))


if __name__ == '__main__':
    unittest.main()
