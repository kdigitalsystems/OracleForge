"""Unit tests for Alpaca order request construction."""
from __future__ import annotations

import unittest
from datetime import date, datetime
from unittest.mock import MagicMock

from alpaca.trading.enums import OrderSide, OrderType, TimeInForce

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


def _api_error(status):
    response = MagicMock(status_code=status)
    return alpaca_client.APIError('{"code": 0, "message": "x"}', MagicMock(response=response))


class FindOrderTests(unittest.TestCase):

    def test_returns_none_only_on_404(self):
        client = MagicMock()
        client.get_order_by_id.side_effect = _api_error(404)
        self.assertIsNone(alpaca_client.find_order(client, 'ord-1'))

    def test_other_errors_propagate(self):
        client = MagicMock()
        client.get_order_by_id.side_effect = _api_error(500)
        with self.assertRaises(alpaca_client.APIError):
            alpaca_client.find_order(client, 'ord-1')


class RecentOrdersTests(unittest.TestCase):

    def test_closed_batch_failure_propagates_instead_of_returning_partial_list(self):
        client = MagicMock()
        client.get_orders.side_effect = [[MagicMock()], RuntimeError('503')]
        with self.assertRaises(RuntimeError):
            alpaca_client.get_all_recent_orders(client)


class PositionDetailsTests(unittest.TestCase):

    def test_position_with_unparseable_price_is_kept(self):
        client = MagicMock()
        client.get_all_positions.return_value = [
            MagicMock(symbol='NVDA', qty='0.02', current_price=None, avg_entry_price='100'),
        ]
        self.assertEqual(
            alpaca_client.get_position_details(client),
            {'NVDA': {'qty': 0.02, 'current_price': None, 'avg_entry_price': 100.0}},
        )


class LastClosedSessionTests(unittest.TestCase):

    def _client(self):
        # Thu 11-26 Thanksgiving (closed), Fri 11-27 early close, Mon 11-30.
        sessions = [
            (date(2026, 11, 25), 16), (date(2026, 11, 27), 13), (date(2026, 11, 30), 16),
        ]
        client = MagicMock()
        client.get_calendar.return_value = [
            MagicMock(date=d, open=datetime(d.year, d.month, d.day, 9, 30),
                      close=datetime(d.year, d.month, d.day, h, 0))
            for d, h in sessions
        ]
        return client

    def test_late_start_and_after_midnight_retry_agree(self):
        client = self._client()
        evening = alpaca_client.last_closed_session(client, datetime(2026, 11, 30, 22, 21))
        after_midnight = alpaca_client.last_closed_session(client, datetime(2026, 12, 1, 0, 13))
        self.assertEqual(evening, date(2026, 11, 30))
        self.assertEqual(after_midnight, date(2026, 11, 30))

    def test_before_the_close_is_the_previous_session(self):
        self.assertEqual(
            alpaca_client.last_closed_session(self._client(), datetime(2026, 11, 30, 8, 30)),
            date(2026, 11, 27),
        )

    def test_weekend_and_holiday_resolve_to_the_last_trading_day(self):
        client = self._client()
        self.assertEqual(alpaca_client.last_closed_session(client, datetime(2026, 11, 29, 19, 0)),
                         date(2026, 11, 27))
        self.assertEqual(alpaca_client.last_closed_session(client, datetime(2026, 11, 26, 19, 0)),
                         date(2026, 11, 25))

    def test_early_close_counts_from_its_own_close_time(self):
        self.assertEqual(
            alpaca_client.last_closed_session(self._client(), datetime(2026, 11, 27, 13, 30)),
            date(2026, 11, 27),
        )

    def test_no_closed_session_raises(self):
        client = MagicMock()
        client.get_calendar.return_value = []
        with self.assertRaises(RuntimeError):
            alpaca_client.last_closed_session(client, datetime(2026, 11, 30, 22, 0))


if __name__ == '__main__':
    unittest.main()
