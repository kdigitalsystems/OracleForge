"""Unit tests for Alpaca order request construction."""
from __future__ import annotations

import unittest
import sys
from datetime import date, datetime
from unittest.mock import MagicMock

from alpaca.trading.enums import OrderSide, OrderType, TimeInForce

# test_trader stubs alpaca_client in sys.modules for pure helper tests. Remove
# that stub here so this module always exercises the real wrapper.
sys.modules.pop('alpaca_client', None)
import alpaca_client


class StopLimitSellTests(unittest.TestCase):

    def test_place_stop_limit_sell_submits_stop_limit_request(self):
        client = MagicMock()
        client.submit_order.return_value.id = 'order-123'

        result = alpaca_client.place_stop_limit_sell(
            client, 'NVDA', qty=0.1234567, stop_price=95.123, limit_price=94.987
        )

        self.assertEqual(result.id, 'order-123')
        client.submit_order.assert_called_once()
        req = client.submit_order.call_args.args[0]
        self.assertEqual(req.symbol, 'NVDA')
        # qty is floored (not rounded) to 6dp so we never request more than held
        self.assertEqual(req.qty, 0.123456)
        self.assertEqual(req.side, OrderSide.SELL)
        self.assertEqual(req.time_in_force, TimeInForce.DAY)
        self.assertEqual(req.type, OrderType.STOP_LIMIT)
        self.assertEqual(req.stop_price, 95.12)
        self.assertEqual(req.limit_price, 94.99)

    def test_place_stop_limit_sell_defaults_limit_to_stop_price(self):
        client = MagicMock()

        alpaca_client.place_stop_limit_sell(client, 'AAPL', qty=0.25, stop_price=10.555)

        req = client.submit_order.call_args.args[0]
        self.assertEqual(req.stop_price, 10.55)
        self.assertEqual(req.limit_price, 10.55)


class SellQtyTests(unittest.TestCase):
    """A 9-dp Alpaca position qty must never round UP past what is held."""

    def test_floors_below_held_amount(self):
        # round() would give 0.164770 (> held); floor must give 0.164769.
        self.assertEqual(alpaca_client._sell_qty(0.164769775), 0.164769)
        self.assertLessEqual(alpaca_client._sell_qty(0.164769775), 0.164769775)

    def test_exact_value_unchanged(self):
        self.assertEqual(alpaca_client._sell_qty(0.05), 0.05)

    def test_limit_sell_uses_floored_qty(self):
        client = MagicMock()
        alpaca_client.place_limit_sell(client, 'NVDA', qty=0.1234567, limit_price=100.0)
        req = client.submit_order.call_args.args[0]
        self.assertEqual(req.qty, 0.123456)


class GetSessionTests(unittest.TestCase):

    def test_returns_open_and_close_for_a_trading_day(self):
        client = MagicMock()
        day = date(2026, 11, 27)
        client.get_calendar.return_value = [MagicMock(
            date=day, open=datetime(2026, 11, 27, 9, 30), close=datetime(2026, 11, 27, 13, 0),
        )]
        self.assertEqual(
            alpaca_client.get_session(client, day),
            (datetime(2026, 11, 27, 9, 30), datetime(2026, 11, 27, 13, 0)),
        )
        req = client.get_calendar.call_args.args[0]
        self.assertEqual((req.start, req.end), (day, day))

    def test_returns_none_on_a_holiday(self):
        client = MagicMock()
        client.get_calendar.return_value = []
        self.assertIsNone(alpaca_client.get_session(client, date(2026, 9, 7)))


if __name__ == '__main__':
    unittest.main()
