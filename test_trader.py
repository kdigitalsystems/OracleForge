"""Unit tests for trader.py helpers (record_buy, record_sell)."""
from __future__ import annotations

import sys
import unittest
from unittest.mock import MagicMock, patch

# Stub heavy dependencies so trader can be imported in a pure-Python test env
sys.modules.setdefault('alpaca_client', MagicMock())
sys.modules.setdefault('pytz', MagicMock())

import trader  # noqa: E402  (must come after stubs)
from trader import record_buy, record_sell  # noqa: E402


# ---------------------------------------------------------------------------
# record_buy
# ---------------------------------------------------------------------------

class RecordBuyTests(unittest.TestCase):

    def _make_meta(self):
        return {}

    @patch('trader.save_json')
    @patch('trader.get_predicting_models', return_value=['model_a'])
    def test_new_position(self, _mock_models, mock_save):
        meta = self._make_meta()
        record_buy(meta, 'NVDA', price=100.0, usd_amount=2.0,
                   pred_date='2026-01-01', buy_high=100.0, sell_low=110.0)

        self.assertIn('NVDA', meta)
        entry = meta['NVDA']
        self.assertEqual(entry['entry_price'], 100.0)
        self.assertEqual(entry['usd_invested'], 2.0)
        self.assertEqual(entry['entry_date'], '2026-01-01')
        self.assertEqual(entry['consensus_buy_high'], 100.0)
        self.assertEqual(entry['consensus_sell_low'], 110.0)
        self.assertEqual(entry['predicting_models'], ['model_a'])
        mock_save.assert_called_once()

    @patch('trader.save_json')
    @patch('trader.get_predicting_models', return_value=[])
    def test_averaging_into_existing_position(self, _mock_models, mock_save):
        # First fill: 2 shares @ $100 = $200 invested
        meta = {
            'NVDA': {
                'entry_price': 100.0,
                'usd_invested': 2.0,
                'entry_date': '2026-01-01',
                'predicting_models': [],
                'consensus_buy_high': 100.0,
                'consensus_sell_low': 110.0,
            }
        }
        # Second fill: $2 @ $110 → avg = (100*2 + 110*2) / 4 = $105
        record_buy(meta, 'NVDA', price=110.0, usd_amount=2.0,
                   pred_date='2026-01-01', buy_high=100.0, sell_low=110.0)

        entry = meta['NVDA']
        self.assertAlmostEqual(entry['entry_price'], 105.0)
        self.assertAlmostEqual(entry['usd_invested'], 4.0)

    @patch('trader.save_json')
    @patch('trader.get_predicting_models', return_value=[])
    def test_usd_amount_is_rounded(self, _mock_models, _mock_save):
        meta = {}
        record_buy(meta, 'AAPL', price=150.0, usd_amount=1.999999,
                   pred_date='2026-01-02', buy_high=150.0, sell_low=160.0)
        # Should be rounded to 4 decimal places
        self.assertEqual(meta['AAPL']['usd_invested'], round(1.999999, 4))


# ---------------------------------------------------------------------------
# record_sell
# ---------------------------------------------------------------------------

class RecordSellTests(unittest.TestCase):

    def _make_meta_with_nvda(self):
        return {
            'NVDA': {
                'entry_price': 100.0,
                'usd_invested': 2.0,
                'entry_date': '2026-01-01',
                'predicting_models': ['model_a'],
                'consensus_buy_high': 100.0,
                'consensus_sell_low': 110.0,
            }
        }

    @patch('trader.save_json')
    def test_win_trade(self, mock_save):
        meta = self._make_meta_with_nvda()
        journal = []
        trade = record_sell(meta, journal, 'NVDA',
                            exit_price=110.0, usd_returned=2.2, close_date='2026-01-02')

        self.assertIsNotNone(trade)
        self.assertEqual(trade['ticker'], 'NVDA')
        self.assertEqual(trade['outcome'], 'win')
        self.assertGreater(trade['pnl_usd'], 0)
        self.assertGreater(trade['pnl_pct'], 0)
        self.assertEqual(trade['exit_price'], 110.0)
        self.assertIn(trade, journal)
        # Ticker should be removed from meta after close
        self.assertNotIn('NVDA', meta)
        # save_json called twice (journal + meta)
        self.assertEqual(mock_save.call_count, 2)

    @patch('trader.save_json')
    def test_loss_trade(self, _mock_save):
        meta = self._make_meta_with_nvda()
        journal = []
        trade = record_sell(meta, journal, 'NVDA',
                            exit_price=96.0, usd_returned=1.92, close_date='2026-01-02')

        self.assertEqual(trade['outcome'], 'loss')
        self.assertLess(trade['pnl_usd'], 0)
        self.assertLess(trade['pnl_pct'], 0)

    @patch('trader.save_json')
    def test_missing_ticker_returns_none(self, mock_save):
        meta = {}
        journal = []
        result = record_sell(meta, journal, 'AAPL',
                             exit_price=200.0, usd_returned=2.0, close_date='2026-01-02')
        self.assertIsNone(result)
        self.assertEqual(journal, [])
        mock_save.assert_not_called()

    @patch('trader.save_json')
    def test_pnl_calculation(self, _mock_save):
        meta = {
            'TSLA': {
                'entry_price': 200.0,
                'usd_invested': 4.0,
                'entry_date': '2026-01-01',
                'predicting_models': [],
                'consensus_buy_high': 200.0,
                'consensus_sell_low': 220.0,
            }
        }
        journal = []
        trade = record_sell(meta, journal, 'TSLA',
                            exit_price=220.0, usd_returned=4.4, close_date='2026-01-03')

        self.assertAlmostEqual(trade['pnl_usd'], 0.4, places=4)
        self.assertAlmostEqual(trade['pnl_pct'], 10.0, places=1)

    @patch('trader.save_json')
    def test_trade_includes_provenance_fields(self, _mock_save):
        meta = self._make_meta_with_nvda()
        journal = []
        trade = record_sell(meta, journal, 'NVDA',
                            exit_price=110.0, usd_returned=2.2, close_date='2026-01-02')

        self.assertEqual(trade['predicting_models'], ['model_a'])
        self.assertEqual(trade['consensus_buy_high'], 100.0)
        self.assertEqual(trade['consensus_sell_low'], 110.0)
        self.assertEqual(trade['entry_date'], '2026-01-01')
        self.assertEqual(trade['close_date'], '2026-01-02')


if __name__ == '__main__':
    unittest.main()
