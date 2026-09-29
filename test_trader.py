"""Unit tests for trader.py helpers (record_buy, record_sell)."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

# Stub heavy dependencies so trader can be imported in a pure-Python test env
sys.modules.setdefault('alpaca_client', MagicMock())
sys.modules.setdefault('pytz', MagicMock())

import trader  # noqa: E402  (must come after stubs)
from trader import get_predicting_models, record_buy, record_sell  # noqa: E402


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
        # Second fill: $2 @ $110. Correct cost basis is share-weighted:
        # shares = 2/100 + 2/110 = 0.038182 ; avg = $4 / 0.038182 = $104.76
        record_buy(meta, 'NVDA', price=110.0, usd_amount=2.0,
                   pred_date='2026-01-01', buy_high=100.0, sell_low=110.0)

        entry = meta['NVDA']
        self.assertAlmostEqual(entry['entry_price'], 104.7619, places=3)
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
# model attribution
# ---------------------------------------------------------------------------

class PredictingModelsTests(unittest.TestCase):

    @patch('trader.load_json')
    def test_fallback_models_are_not_attributed_to_trade(self, mock_load):
        mock_load.return_value = {
            'NVDA': {
                'models': {
                    'real_model': {'sell_low': 110.0},
                    'fallback_model': {'sell_low': 111.0, 'fallback': True},
                    'skipped_model': {'skipped': True},
                }
            }
        }

        self.assertEqual(get_predicting_models('NVDA', '2026-01-01'), ['real_model'])


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

    @patch('trader.save_json')
    def test_partial_sell_keeps_remaining_position(self, mock_save):
        meta = self._make_meta_with_nvda()
        journal = []
        trade = record_sell(
            meta, journal, 'NVDA',
            exit_price=110.0, usd_returned=1.1, close_date='2026-01-02',
            close_fraction=0.5,
        )

        self.assertIsNotNone(trade)
        self.assertEqual(trade['usd_invested'], 1.0)
        self.assertAlmostEqual(trade['pnl_usd'], 0.1, places=4)
        self.assertIn('NVDA', meta)
        self.assertEqual(meta['NVDA']['usd_invested'], 1.0)
        self.assertEqual(mock_save.call_count, 2)

    @patch('trader.save_json')
    def test_zero_fraction_sell_is_ignored(self, mock_save):
        meta = self._make_meta_with_nvda()
        journal = []
        trade = record_sell(
            meta, journal, 'NVDA',
            exit_price=110.0, usd_returned=0.0, close_date='2026-01-02',
            close_fraction=0.0,
        )

        self.assertIsNone(trade)
        self.assertIn('NVDA', meta)
        self.assertEqual(journal, [])
        mock_save.assert_not_called()


# ---------------------------------------------------------------------------
# run_open order placement
# ---------------------------------------------------------------------------

class RunOpenTests(unittest.TestCase):

    def setUp(self):
        trader.alpaca_client.reset_mock()

    @patch('trader.time.sleep')
    @patch('trader.now_et', return_value='2026-01-02T09:30:00-05:00')
    @patch('trader.today_str', return_value='2026-01-02')
    @patch('trader.save_json')
    @patch('trader.load_todays_signals')
    @patch('trader.load_json')
    def test_reprotects_existing_position_with_real_stop_limit_order(
        self,
        mock_load_json,
        mock_load_signals,
        _mock_save,
        _mock_today,
        _mock_now,
        _mock_sleep,
    ):
        mock_load_signals.return_value = (
            [{'ticker': 'NVDA', 'buy_high': 100.0, 'sell_low': 110.0, 'upside_pct': 10.0}],
            '2026-01-01',
        )
        mock_load_json.side_effect = [
            {'max_per_trade_usd': 2.0, 'max_position_usd': 8.0, 'stop_loss_pct': 0.95},
            {},  # open_orders
            {
                'NVDA': {
                    'entry_price': 100.0,
                    'usd_invested': 2.0,
                    'entry_date': '2026-01-01',
                    'predicting_models': ['model_a'],
                    'consensus_buy_high': 100.0,
                    'consensus_sell_low': 110.0,
                }
            },
        ]

        client = MagicMock()
        client.get_account.return_value.buying_power = 100.0
        trader.alpaca_client.get_trading_client.return_value = client
        trader.alpaca_client.get_positions.return_value = {'NVDA': 2.0}
        trader.alpaca_client.get_position_qty.return_value = 0.02
        trader.alpaca_client.place_limit_sell.return_value.id = 'profit-order'
        trader.alpaca_client.place_stop_limit_sell.return_value.id = 'stop-order'

        trader.run_open(dry_run=False)

        # Profit-target sell placed; stop is NOT a resting order (enforced at --close).
        trader.alpaca_client.place_limit_sell.assert_called_once_with(client, 'NVDA', 0.02, 110.0)
        trader.alpaca_client.place_stop_limit_sell.assert_not_called()

    @patch('trader.time.sleep')
    @patch('trader.now_et', return_value='2026-01-02T09:30:00-05:00')
    @patch('trader.today_str', return_value='2026-01-02')
    @patch('trader.save_json')
    @patch('trader.load_todays_signals', return_value=([], None))
    @patch('trader.load_json')
    def test_sell_replacement_scoped_to_tracked_stake_on_shared_account(
        self, mock_load_json, _mock_signals, _mock_save, _mock_today, _mock_now, _mock_sleep,
    ):
        # This paper account is shared with other trading systems: the
        # account really holds 0.10 NVDA, but OracleForge only bought 0.02
        # (usd_invested 2.0 / entry_price 100.0). The morning profit-target
        # re-placement must reserve only OUR 0.02, not sweep the other
        # system's 0.08 into our sell (how a foreign PLTR stake got sold on
        # 2026-08-12).
        mock_load_json.side_effect = [
            {'max_per_trade_usd': 2.0, 'max_position_usd': 8.0, 'stop_loss_pct': 0.95},
            {},  # open_orders
            {
                'NVDA': {
                    'entry_price': 100.0,
                    'usd_invested': 2.0,
                    'entry_date': '2026-01-01',
                    'predicting_models': ['model_a'],
                    'consensus_buy_high': 100.0,
                    'consensus_sell_low': 110.0,
                }
            },
        ]
        client = MagicMock()
        client.get_account.return_value.buying_power = 100.0
        trader.alpaca_client.get_trading_client.return_value = client
        trader.alpaca_client.get_positions.return_value = {'NVDA': 10.0}
        trader.alpaca_client.get_position_qty.return_value = 0.10  # whole account position
        trader.alpaca_client.place_limit_sell.return_value.id = 'profit-order'

        trader.run_open(dry_run=False)

        trader.alpaca_client.place_limit_sell.assert_called_once_with(client, 'NVDA', 0.02, 110.0)

    @patch('trader.time.sleep')
    @patch('trader.now_et', return_value='2026-01-02T09:30:00-05:00')
    @patch('trader.today_str', return_value='2026-01-02')
    @patch('trader.save_json')
    @patch('trader.load_todays_signals')
    @patch('trader.load_json')
    def test_protect_only_places_sells_but_never_buys(
        self, mock_load_json, mock_load_signals, _mock_save, _mock_today, _mock_now, _mock_sleep,
    ):
        # protect_only exists for mornings after a missed/failed nightly: no
        # trusted signals, so no buys, but held positions must still get
        # their profit-target sells re-placed. Signals must not even be
        # loaded (they would be stale).
        mock_load_json.side_effect = [
            {'max_per_trade_usd': 2.0, 'max_position_usd': 8.0, 'stop_loss_pct': 0.95},
            {},  # open_orders
            {
                'NVDA': {
                    'entry_price': 100.0,
                    'usd_invested': 2.0,
                    'entry_date': '2026-01-01',
                    'predicting_models': ['model_a'],
                    'consensus_buy_high': 100.0,
                    'consensus_sell_low': 110.0,
                }
            },
        ]
        client = MagicMock()
        client.get_account.return_value.buying_power = 100.0
        trader.alpaca_client.get_trading_client.return_value = client
        trader.alpaca_client.get_positions.return_value = {'NVDA': 2.0}
        trader.alpaca_client.get_position_qty.return_value = 0.02
        trader.alpaca_client.place_limit_sell.return_value.id = 'profit-order'

        trader.run_open(dry_run=False, protect_only=True)

        mock_load_signals.assert_not_called()
        trader.alpaca_client.place_limit_buy.assert_not_called()
        trader.alpaca_client.place_limit_sell.assert_called_once_with(client, 'NVDA', 0.02, 110.0)

    @patch('trader.time.sleep')
    @patch('trader.now_et', return_value='2026-01-02T09:30:00-05:00')
    @patch('trader.today_str', return_value='2026-01-02')
    @patch('trader.save_json')
    @patch('trader.load_todays_signals', return_value=([], None))
    @patch('trader.load_json')
    def test_protects_existing_positions_even_with_no_active_signals(
        self,
        mock_load_json,
        _mock_load_signals,
        _mock_save,
        _mock_today,
        _mock_now,
        _mock_sleep,
    ):
        mock_load_json.side_effect = [
            {'max_per_trade_usd': 2.0, 'max_position_usd': 8.0, 'stop_loss_pct': 0.95},
            {},  # open_orders
            {
                'NVDA': {
                    'entry_price': 100.0,
                    'usd_invested': 2.0,
                    'entry_date': '2026-01-01',
                    'predicting_models': ['model_a'],
                    'consensus_buy_high': 100.0,
                    'consensus_sell_low': 110.0,
                }
            },
        ]

        client = MagicMock()
        client.get_account.return_value.buying_power = 100.0
        trader.alpaca_client.get_trading_client.return_value = client
        trader.alpaca_client.get_positions.return_value = {'NVDA': 2.0}
        trader.alpaca_client.get_position_qty.return_value = 0.02
        trader.alpaca_client.place_limit_sell.return_value.id = 'profit-order'
        trader.alpaca_client.place_stop_limit_sell.return_value.id = 'stop-order'

        trader.run_open(dry_run=False)

        # Existing position still gets its profit-target sell even with no new
        # signals; the stop is enforced at --close, not as a resting order.
        trader.alpaca_client.place_limit_sell.assert_called_once_with(client, 'NVDA', 0.02, 110.0)
        trader.alpaca_client.place_stop_limit_sell.assert_not_called()


# ---------------------------------------------------------------------------
# get_predicting_models
# ---------------------------------------------------------------------------

class GetPredictingModelsTests(unittest.TestCase):

    @patch('trader.load_json')
    def test_excludes_fallback_and_skipped_models(self, mock_load_json):
        # One real model, one fallback (synthetic), one earnings-skipped entry.
        mock_load_json.return_value = {
            'NVDA': {
                'models': {
                    'real_model':     {'buy_low': 98, 'buy_high': 100, 'sell_low': 110, 'sell_high': 112},
                    'fallback_model': {'buy_low': 98, 'buy_high': 100, 'sell_low': 110, 'sell_high': 112,
                                       'fallback': True},
                    'skipped_model':  {'skipped': True, 'reason': 'upcoming_earnings'},
                }
            }
        }
        # Only the real model drove the consensus, so only it should be credited.
        self.assertEqual(trader.get_predicting_models('NVDA', '2026-01-01'), ['real_model'])

    @patch('trader.load_json', return_value={})
    def test_missing_ticker_returns_empty(self, _mock_load_json):
        self.assertEqual(trader.get_predicting_models('ZZZZ', '2026-01-01'), [])


# ---------------------------------------------------------------------------
# run_close end-of-day stop check
# ---------------------------------------------------------------------------

class RunCloseStopTests(unittest.TestCase):

    def setUp(self):
        trader.alpaca_client.reset_mock()

    def _held_state(self, entry_date='2026-01-01', max_hold_days=15):
        # open_orders with a held NVDA that has no resting orders to settle,
        # plus positions_meta for it. Returned as load_json side_effect inputs.
        meta = {'NVDA': {'entry_price': 100.0, 'usd_invested': 2.0, 'entry_date': entry_date,
                         'predicting_models': ['m'], 'consensus_buy_high': 100.0,
                         'consensus_sell_low': 110.0}}
        journal = []
        side_effect = [
            {'max_per_trade_usd': 2.0, 'max_position_usd': 8.0, 'stop_loss_pct': 0.95,
             'max_hold_days': max_hold_days},  # config
            {'NVDA': {'buy_order_id': None, 'sell_order_id': None, 'stop_order_id': None}},  # open_orders
            meta,        # positions_meta
            journal,     # journal
        ]
        return meta, journal, side_effect

    @patch('trader.time.sleep')
    @patch('trader.now_et', return_value='2026-01-02T16:05:00-05:00')
    @patch('trader.today_str', return_value='2026-01-02')
    @patch('trader.save_json')
    @patch('trader.load_json')
    def test_eod_stop_sells_position_below_stop(self, mock_load_json, _save, _today, _now, _sleep):
        meta, journal, mock_load_json.side_effect = self._held_state()
        client = MagicMock()
        trader.alpaca_client.get_trading_client.return_value = client
        trader.alpaca_client.get_all_recent_orders.return_value = []
        # 94 < 100 * 0.95 = 95 -> stop triggers
        trader.alpaca_client.get_position_details.return_value = {
            'NVDA': {'qty': 0.02, 'current_price': 94.0, 'avg_entry_price': 100.0}
        }

        trader.run_close(dry_run=False)

        trader.alpaca_client.place_market_sell.assert_called_once_with(client, 'NVDA', 0.02)
        self.assertEqual(len(journal), 1)
        self.assertEqual(journal[0]['outcome'], 'loss')
        self.assertNotIn('NVDA', meta)  # position closed out

    @patch('trader.time.sleep')
    @patch('trader.now_et', return_value='2026-01-02T16:05:00-05:00')
    @patch('trader.today_str', return_value='2026-01-02')
    @patch('trader.save_json')
    @patch('trader.load_json')
    def test_eod_stop_holds_position_above_stop(self, mock_load_json, _save, _today, _now, _sleep):
        meta, journal, mock_load_json.side_effect = self._held_state()
        client = MagicMock()
        trader.alpaca_client.get_trading_client.return_value = client
        trader.alpaca_client.get_all_recent_orders.return_value = []
        # 98 > 95 -> no stop
        trader.alpaca_client.get_position_details.return_value = {
            'NVDA': {'qty': 0.02, 'current_price': 98.0, 'avg_entry_price': 100.0}
        }

        trader.run_close(dry_run=False)

        trader.alpaca_client.place_market_sell.assert_not_called()
        self.assertEqual(journal, [])
        self.assertIn('NVDA', meta)

    @patch('trader.time.sleep')
    @patch('trader.now_et', return_value='2026-01-02T16:05:00-05:00')
    @patch('trader.today_str', return_value='2026-01-02')
    @patch('trader.save_json')
    @patch('trader.load_json')
    def test_eod_stop_caps_return_when_real_qty_exceeds_tracked(self, mock_load_json, _save, _today, _now, _sleep):
        # Tracked: 0.02 shares (usd_invested 2.0 / entry_price 100.0). Alpaca
        # actually holds 0.06 (e.g. a duplicate buy from a workflow retry) --
        # the extra 0.04 must not be counted as profit.
        meta, journal, mock_load_json.side_effect = self._held_state()
        client = MagicMock()
        trader.alpaca_client.get_trading_client.return_value = client
        trader.alpaca_client.get_all_recent_orders.return_value = []
        trader.alpaca_client.get_position_details.return_value = {
            'NVDA': {'qty': 0.06, 'current_price': 94.0, 'avg_entry_price': 100.0}
        }

        trader.run_close(dry_run=False)

        trader.alpaca_client.place_market_sell.assert_called_once_with(client, 'NVDA', 0.02)
        self.assertEqual(len(journal), 1)
        # Capped to the tracked 0.02 shares, not the real 0.06.
        self.assertEqual(journal[0]['usd_returned'], round(94.0 * 0.02, 4))
        self.assertAlmostEqual(journal[0]['pnl_usd'], round(94.0 * 0.02, 4) - 2.0, places=4)
        self.assertNotIn('NVDA', meta)

    @patch('trader.time.sleep')
    @patch('trader.now_et', return_value='2026-01-02T16:05:00-05:00')
    @patch('trader.today_str', return_value='2026-01-02')
    @patch('trader.save_json')
    @patch('trader.load_json')
    def test_eod_stop_caps_return_when_real_qty_below_tracked(self, mock_load_json, _save, _today, _now, _sleep):
        # Tracked: 0.02 shares. Alpaca actually only holds 0.0066667 (a third)
        # -- the tracked cost basis is treated as fully at risk since the
        # missing shares can't be recovered, rather than being papered over.
        meta, journal, mock_load_json.side_effect = self._held_state()
        client = MagicMock()
        trader.alpaca_client.get_trading_client.return_value = client
        trader.alpaca_client.get_all_recent_orders.return_value = []
        trader.alpaca_client.get_position_details.return_value = {
            'NVDA': {'qty': 0.0066667, 'current_price': 94.0, 'avg_entry_price': 100.0}
        }

        trader.run_close(dry_run=False)

        # Only the real 0.0066667 shares exist to sell.
        trader.alpaca_client.place_market_sell.assert_called_once_with(client, 'NVDA', 0.0066667)
        self.assertEqual(len(journal), 1)
        self.assertEqual(journal[0]['usd_returned'], round(94.0 * 0.0066667, 4))
        self.assertNotIn('NVDA', meta)

    @patch('trader.time.sleep')
    @patch('trader.now_et', return_value='2026-01-20T16:05:00-05:00')
    @patch('trader.today_str', return_value='2026-01-20')
    @patch('trader.save_json')
    @patch('trader.load_json')
    def test_max_hold_days_force_closes_stale_position(self, mock_load_json, _save, _today, _now, _sleep):
        # Entered 2026-01-01, closing 2026-01-20 -> held 19 days >= default max of 15.
        meta, journal, mock_load_json.side_effect = self._held_state(entry_date='2026-01-01')
        client = MagicMock()
        trader.alpaca_client.get_trading_client.return_value = client
        trader.alpaca_client.get_all_recent_orders.return_value = []
        # Price is between stop (95) and target (110) -- no stop, no fill, just stale.
        trader.alpaca_client.get_position_details.return_value = {
            'NVDA': {'qty': 0.02, 'current_price': 102.0, 'avg_entry_price': 100.0}
        }

        trader.run_close(dry_run=False)

        trader.alpaca_client.place_market_sell.assert_called_once_with(client, 'NVDA', 0.02)
        self.assertEqual(len(journal), 1)
        self.assertNotIn('NVDA', meta)

    @patch('trader.time.sleep')
    @patch('trader.now_et', return_value='2026-01-10T16:05:00-05:00')
    @patch('trader.today_str', return_value='2026-01-10')
    @patch('trader.save_json')
    @patch('trader.load_json')
    def test_holds_position_under_max_hold_days(self, mock_load_json, _save, _today, _now, _sleep):
        # Entered 2026-01-01, closing 2026-01-10 -> held 9 days < default max of 15.
        meta, journal, mock_load_json.side_effect = self._held_state(entry_date='2026-01-01')
        client = MagicMock()
        trader.alpaca_client.get_trading_client.return_value = client
        trader.alpaca_client.get_all_recent_orders.return_value = []
        trader.alpaca_client.get_position_details.return_value = {
            'NVDA': {'qty': 0.02, 'current_price': 102.0, 'avg_entry_price': 100.0}
        }

        trader.run_close(dry_run=False)

        trader.alpaca_client.place_market_sell.assert_not_called()
        self.assertEqual(journal, [])
        self.assertIn('NVDA', meta)

    @patch('trader.time.sleep')
    @patch('trader.now_et', return_value='2026-01-20T16:05:00-05:00')
    @patch('trader.today_str', return_value='2026-01-20')
    @patch('trader.save_json')
    @patch('trader.load_json')
    def test_orphaned_position_removed_when_alpaca_shows_no_real_shares(
        self, mock_load_json, _save, _today, _now, _sleep
    ):
        # Entered 2026-01-01, closing 2026-01-20 -> held 19 days >= default max
        # of 15, and Alpaca reports no real position at all for NVDA.
        meta, journal, mock_load_json.side_effect = self._held_state(entry_date='2026-01-01')
        client = MagicMock()
        trader.alpaca_client.get_trading_client.return_value = client
        trader.alpaca_client.get_all_recent_orders.return_value = []
        trader.alpaca_client.get_position_details.return_value = {}

        trader.run_close(dry_run=False)

        # No real shares to sell -- must not attempt a market sell or fabricate
        # a P&L entry, but the ghost tracking entry must be cleared so it
        # doesn't block forever.
        trader.alpaca_client.place_market_sell.assert_not_called()
        self.assertEqual(journal, [])
        self.assertNotIn('NVDA', meta)

        # The pop must actually be persisted -- record_sell() isn't called on
        # this path (no P&L to record), so nothing else writes
        # positions_meta.json here. Regression check: this previously popped
        # only the in-memory dict, silently reverting every run.
        saved_meta_calls = [
            call.args[1] for call in _save.call_args_list
            if call.args[0] == trader.POSITIONS_META_FILE
        ]
        self.assertTrue(saved_meta_calls, "positions_meta.json was never saved")
        self.assertNotIn('NVDA', saved_meta_calls[-1])

    @patch('trader.time.sleep')
    @patch('trader.now_et', return_value='2026-01-10T16:05:00-05:00')
    @patch('trader.today_str', return_value='2026-01-10')
    @patch('trader.save_json')
    @patch('trader.load_json')
    def test_missing_position_kept_when_under_max_hold_days(self, mock_load_json, _save, _today, _now, _sleep):
        # Entered 2026-01-01, closing 2026-01-10 -> held 9 days < default max
        # of 15. Alpaca showing no position yet could just be a transient API
        # gap this early -- don't remove tracking prematurely.
        meta, journal, mock_load_json.side_effect = self._held_state(entry_date='2026-01-01')
        client = MagicMock()
        trader.alpaca_client.get_trading_client.return_value = client
        trader.alpaca_client.get_all_recent_orders.return_value = []
        trader.alpaca_client.get_position_details.return_value = {}

        trader.run_close(dry_run=False)

        trader.alpaca_client.place_market_sell.assert_not_called()
        self.assertEqual(journal, [])
        self.assertIn('NVDA', meta)


# ---------------------------------------------------------------------------
# run_close: normal sell-fill / stop-fill settlement (previously untested --
# this is the path that produced the PLTR/ORCL 200-400%+ fantasy "wins" on
# 2026-07-27/07-30, which the EOD-stop/max-hold fix from the same bug class
# did not cover)
# ---------------------------------------------------------------------------

class RunCloseFillTests(unittest.TestCase):

    def setUp(self):
        trader.alpaca_client.reset_mock()

    def _fill_state(self, order_qty, order_id_field='sell_order_id', order_id='ord-1',
                     tracked_usd_invested=2.0, tracked_entry_price=100.0):
        meta = {'NVDA': {'entry_price': tracked_entry_price, 'usd_invested': tracked_usd_invested,
                         'entry_date': '2026-01-01', 'predicting_models': ['m'],
                         'consensus_buy_high': 100.0, 'consensus_sell_low': 110.0}}
        journal = []
        open_orders = {'NVDA': {
            'buy_order_id': None, 'sell_order_id': None, 'stop_order_id': None,
            'qty': order_qty, 'sell_limit': 110.0, 'buy_limit': 100.0, 'stop_limit': 95.0,
        }}
        open_orders['NVDA'][order_id_field] = order_id
        side_effect = [
            {'max_per_trade_usd': 2.0, 'max_position_usd': 8.0, 'stop_loss_pct': 0.95, 'max_hold_days': 15},
            open_orders,
            meta,
            journal,
        ]
        return meta, journal, side_effect

    @patch('trader.time.sleep')
    @patch('trader.now_et', return_value='2026-01-05T16:05:00-05:00')
    @patch('trader.today_str', return_value='2026-01-05')
    @patch('trader.save_json')
    @patch('trader.load_json')
    def test_sell_fill_caps_return_when_real_fill_exceeds_tracked_qty(self, mock_load_json, _save, _today, _now, _sleep):
        # Tracked: 0.02 shares (usd_invested 2.0 / entry_price 100.0). The DAY
        # sell order that actually filled sold 0.06 real shares -- e.g.
        # re-placed the next morning at a since-diverged real Alpaca qty. The
        # extra 0.04 shares must not be counted as profit.
        meta, journal, mock_load_json.side_effect = self._fill_state(order_qty=0.02)
        client = MagicMock()
        trader.alpaca_client.get_trading_client.return_value = client

        order = MagicMock()
        order.id = 'ord-1'
        order.status = 'filled'
        order.filled_avg_price = 110.0
        order.filled_qty = 0.06
        trader.alpaca_client.get_all_recent_orders.return_value = [order]

        trader.run_close(dry_run=False)

        self.assertEqual(len(journal), 1)
        self.assertEqual(journal[0]['usd_returned'], round(110.0 * 0.02, 4))
        self.assertAlmostEqual(journal[0]['pnl_usd'], round(110.0 * 0.02, 4) - 2.0, places=4)
        self.assertLess(journal[0]['pnl_pct'], 50)  # sanity: not a 230%+ fantasy return
        self.assertNotIn('NVDA', meta)

    @patch('trader.time.sleep')
    @patch('trader.now_et', return_value='2026-01-05T16:05:00-05:00')
    @patch('trader.today_str', return_value='2026-01-05')
    @patch('trader.save_json')
    @patch('trader.load_json')
    def test_stop_fill_caps_return_when_real_fill_exceeds_tracked_qty(self, mock_load_json, _save, _today, _now, _sleep):
        # Same desync, but through the (currently dormant, kept for reference)
        # stop-limit fill handler -- same bug pattern, same fix.
        meta, journal, mock_load_json.side_effect = self._fill_state(
            order_qty=0.02, order_id_field='stop_order_id',
        )
        client = MagicMock()
        trader.alpaca_client.get_trading_client.return_value = client

        order = MagicMock()
        order.id = 'ord-1'
        order.status = 'filled'
        order.filled_avg_price = 95.0
        order.filled_qty = 0.06
        trader.alpaca_client.get_all_recent_orders.return_value = [order]

        trader.run_close(dry_run=False)

        self.assertEqual(len(journal), 1)
        self.assertEqual(journal[0]['usd_returned'], round(95.0 * 0.02, 4))
        self.assertNotIn('NVDA', meta)

    @patch('trader.time.sleep')
    @patch('trader.now_et', return_value='2026-01-05T16:05:00-05:00')
    @patch('trader.today_str', return_value='2026-01-05')
    @patch('trader.save_json')
    @patch('trader.load_json')
    def test_genuinely_complete_fill_pops_position_despite_float_rounding(
        self, mock_load_json, _save, _today, _now, _sleep
    ):
        # Regression for a real production incident (BP/CHWY/CPRT/FCX/LYFT/
        # PBR/TOST, all closed correctly in the journal 2026-08-14..08-20 but
        # left behind as $0 zombie entries in positions_meta for weeks).
        # positions_meta['usd_invested'] / ['entry_price'] is an
        # independently-computed float via division; it essentially never
        # equals the real order/fill qty bit-for-bit even for a fully
        # complete fill. Using it as the close_fraction denominator (as an
        # earlier draft of the qty-desync fix did) meant close_fraction
        # landed at ~0.9999916 instead of 1.0 -- just under the
        # >=0.999999 "fully closed" threshold -- so every complete sell
        # was mistaken for a partial one and never popped.
        entry_price = 42.608
        usd_invested = 1.9818
        implied_qty = usd_invested / entry_price  # 0.046512392039053704
        real_qty = round(implied_qty, 6)           # 0.046512 -- what was actually bought/sold

        meta, journal, mock_load_json.side_effect = self._fill_state(
            order_qty=real_qty, tracked_usd_invested=usd_invested, tracked_entry_price=entry_price,
        )
        client = MagicMock()
        trader.alpaca_client.get_trading_client.return_value = client

        order = MagicMock()
        order.id = 'ord-1'
        order.status = 'filled'
        order.filled_avg_price = 44.012
        order.filled_qty = real_qty  # exactly what was ordered -- a complete fill

        trader.alpaca_client.get_all_recent_orders.return_value = [order]

        trader.run_close(dry_run=False)

        self.assertEqual(len(journal), 1)
        self.assertEqual(journal[0]['outcome'], 'win')
        # The position must be gone, not left behind as a $0 zombie entry.
        self.assertNotIn('NVDA', meta)

    def _run_exit_fill(self, order_id_field, tracked_qty, filled_qty):
        meta, journal, side_effect = self._fill_state(
            order_qty=tracked_qty, order_id_field=order_id_field,
        )
        open_orders = side_effect[1]
        mock_load_json = patch('trader.load_json', side_effect=side_effect).start()
        self.addCleanup(patch.stopall)
        for target in ('trader.save_json', 'trader.time.sleep'):
            patch(target).start()
        patch('trader.today_str', return_value='2026-01-05').start()
        patch('trader.now_et', return_value='2026-01-05T16:05:00-05:00').start()
        trader.alpaca_client.get_trading_client.return_value = MagicMock()
        trader.alpaca_client.get_position_details.return_value = {
            'NVDA': {'qty': max(tracked_qty - filled_qty, 0.0), 'current_price': 105.0},
        }

        order = MagicMock()
        order.id = 'ord-1'
        order.status = 'filled'
        order.filled_avg_price = 110.0
        order.filled_qty = filled_qty
        trader.alpaca_client.get_all_recent_orders.return_value = [order]

        trader.run_close(dry_run=False)
        self.assertEqual(mock_load_json.call_count, 4)
        return meta, journal, open_orders

    def test_partial_sell_fill_keeps_remainder_without_a_buy_in_the_same_run(self):
        # Regression: the partial-exit branch read total_qty, which is only
        # assigned when a buy fills in the same run, so a partial sell on its
        # own raised UnboundLocalError after record_sell had already run.
        meta, journal, open_orders = self._run_exit_fill('sell_order_id', 0.02, 0.005)

        self.assertEqual(len(journal), 1)
        self.assertIn('NVDA', meta)
        self.assertAlmostEqual(open_orders['NVDA']['qty'], 0.015)
        self.assertIsNone(open_orders['NVDA']['sell_order_id'])

    def test_partial_stop_fill_keeps_remainder_without_a_buy_in_the_same_run(self):
        meta, journal, open_orders = self._run_exit_fill('stop_order_id', 0.02, 0.005)

        self.assertEqual(len(journal), 1)
        self.assertAlmostEqual(open_orders['NVDA']['qty'], 0.015)
        self.assertIsNone(open_orders['NVDA']['stop_order_id'])

    def test_fill_of_floored_sell_qty_is_a_full_exit(self):
        # Alpaca reports positions to 9 dp, but sells are placed for the qty
        # floored to 6 dp, so a complete fill leaves unsellable dust. That
        # must close the position, not be treated as a partial exit.
        meta, journal, _ = self._run_exit_fill('sell_order_id', 0.015721987, 0.015721)

        self.assertEqual(len(journal), 1)
        self.assertNotIn('NVDA', meta)

    def test_fill_of_exact_6dp_qty_hit_by_float_floor_is_a_full_exit(self):
        # floor(0.001001 * 1e6) == 1000: the placed sell is one microshare
        # short of the tracked qty purely from float error.
        meta, journal, _ = self._run_exit_fill('sell_order_id', 0.001001, 0.001)

        self.assertEqual(len(journal), 1)
        self.assertNotIn('NVDA', meta)


class SessionGateTests(unittest.TestCase):
    """The crons are fixed UTC, so they drift against the market at each DST
    change (20:05 UTC was 3:05 PM EST, mid-session). The gate keys off
    Alpaca's calendar instead."""

    REGULAR = (datetime(2026, 11, 2, 9, 30), datetime(2026, 11, 2, 16, 0))
    EARLY_CLOSE = (datetime(2026, 11, 27, 9, 30), datetime(2026, 11, 27, 13, 0))

    def test_close_skipped_while_session_is_open(self):
        reason = trader.session_skip_reason('close', self.REGULAR, datetime(2026, 11, 2, 15, 5))
        self.assertIn('still open', reason)

    def test_close_proceeds_after_close_including_early_close(self):
        self.assertIsNone(trader.session_skip_reason('close', self.REGULAR, datetime(2026, 11, 2, 16, 5)))
        self.assertIsNone(trader.session_skip_reason('close', self.EARLY_CLOSE, datetime(2026, 11, 27, 16, 5)))

    def test_open_proceeds_before_and_during_session(self):
        self.assertIsNone(trader.session_skip_reason('open', self.REGULAR, datetime(2026, 11, 2, 8, 30)))
        self.assertIsNone(trader.session_skip_reason('open', self.REGULAR, datetime(2026, 11, 2, 11, 0)))

    def test_open_skipped_once_session_closed(self):
        reason = trader.session_skip_reason('open', self.EARLY_CLOSE, datetime(2026, 11, 27, 14, 0))
        self.assertIn('already closed', reason)

    def test_both_skipped_on_non_trading_day(self):
        for mode in ('open', 'close'):
            self.assertIn('market closed', trader.session_skip_reason(mode, None, datetime(2026, 9, 7, 16, 5)))


class MainSessionGateTests(unittest.TestCase):

    def setUp(self):
        trader.alpaca_client.reset_mock()
        patch('trader.ET', timezone.utc).start()
        self.addCleanup(patch.stopall)

    def _main(self, *argv):
        with patch.object(sys, 'argv', ['trader.py', *argv]):
            trader.main()

    def test_skipped_run_does_not_trade_and_flags_the_workflow(self):
        trader.alpaca_client.get_session.return_value = None
        run_close = patch('trader.run_close').start()
        with tempfile.NamedTemporaryFile('r', suffix='.out') as out:
            with patch.dict(os.environ, {'GITHUB_OUTPUT': out.name}):
                self._main('--close')
            self.assertEqual(out.read(), 'skipped=true\n')
        run_close.assert_not_called()

    def test_force_bypasses_the_session_check(self):
        run_open = patch('trader.run_open').start()
        self._main('--open', '--force')
        trader.alpaca_client.get_session.assert_not_called()
        run_open.assert_called_once_with(dry_run=False, protect_only=False)


if __name__ == '__main__':
    unittest.main()
