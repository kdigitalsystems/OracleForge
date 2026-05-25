"""Unit tests for backtest helpers"""
import unittest

from backtest import _finalize_bucket, _new_bucket, _update_bucket, simulate_range_outcome


RANGE_WIN = {'buy_high': 100.0, 'sell_low': 107.0}
RANGE_STOP = {'buy_high': 100.0, 'sell_low': 110.0}
RANGE_MISS = {'buy_high': 100.0, 'sell_low': 110.0}
RANGE_NO_TRIGGER = {'buy_high': 95.0, 'sell_low': 105.0}


class SimulateRangeTests(unittest.TestCase):
    def test_win(self):
        bar = {'open': 101, 'high': 108, 'low': 99, 'close': 106}  # low=99 > stop(98)
        result = simulate_range_outcome(bar, RANGE_WIN)
        self.assertEqual(result['outcome'], 'win')
        self.assertTrue(result['triggered'])
        self.assertGreater(result['return_pct'], 0)

    def test_stop(self):
        bar = {'open': 101, 'high': 101, 'low': 97, 'close': 99}
        result = simulate_range_outcome(bar, RANGE_STOP)
        self.assertEqual(result['outcome'], 'stop')
        self.assertEqual(result['return_pct'], -2.0)
        self.assertTrue(result['triggered'])

    def test_miss(self):
        bar = {'open': 101, 'high': 104, 'low': 99, 'close': 103}
        result = simulate_range_outcome(bar, RANGE_MISS)
        self.assertEqual(result['outcome'], 'miss')
        self.assertTrue(result['triggered'])

    def test_no_trigger(self):
        bar = {'open': 101, 'high': 108, 'low': 97, 'close': 105}
        result = simulate_range_outcome(bar, RANGE_NO_TRIGGER)
        self.assertEqual(result['outcome'], 'no_trigger')
        self.assertFalse(result['triggered'])


class BucketTests(unittest.TestCase):
    def test_update_and_finalize(self):
        bucket = _new_bucket()
        _update_bucket(bucket, simulate_range_outcome({'open': 101, 'high': 108, 'low': 99, 'close': 106}, RANGE_WIN))
        _update_bucket(bucket, simulate_range_outcome({'open': 101, 'high': 101, 'low': 97, 'close': 99}, RANGE_STOP))
        final = _finalize_bucket(bucket)
        self.assertEqual(final['trades'], 2)
        self.assertEqual(final['triggered'], 2)
        self.assertEqual(final['wins'], 1)
        self.assertEqual(final['stops'], 1)
        self.assertAlmostEqual(final['win_rate'], 0.5)

    def test_no_trigger_not_counted(self):
        bucket = _new_bucket()
        _update_bucket(bucket, simulate_range_outcome({'open': 101, 'high': 108, 'low': 97, 'close': 105}, RANGE_NO_TRIGGER))
        final = _finalize_bucket(bucket)
        self.assertEqual(final['trades'], 1)
        self.assertEqual(final['triggered'], 0)
        self.assertEqual(final['win_rate'], 0.0)

    def test_finalize_empty(self):
        final = _finalize_bucket(_new_bucket())
        self.assertEqual(final['trades'], 0)
        self.assertEqual(final['win_rate'], 0.0)

    # ----- Risk metric tests -----

    def test_finalize_includes_risk_fields(self):
        bucket = _new_bucket()
        _update_bucket(bucket, simulate_range_outcome({'open': 101, 'high': 108, 'low': 99, 'close': 106}, RANGE_WIN))
        final = _finalize_bucket(bucket)
        for field in ('avg_win_pct', 'avg_loss_pct', 'profit_factor', 'max_consecutive_losses'):
            self.assertIn(field, final, f"Missing risk field: {field}")

    def test_profit_factor_one_win_one_stop(self):
        bucket = _new_bucket()
        # Win returns 7%, stop returns -2%
        _update_bucket(bucket, simulate_range_outcome({'open': 101, 'high': 108, 'low': 99, 'close': 106}, RANGE_WIN))
        _update_bucket(bucket, simulate_range_outcome({'open': 101, 'high': 101, 'low': 97, 'close': 99}, RANGE_STOP))
        final = _finalize_bucket(bucket)
        # gross_profit = ~7%, gross_loss = 2%  => pf = ~3.5
        self.assertIsNotNone(final['profit_factor'])
        self.assertGreater(final['profit_factor'], 1.0)

    def test_profit_factor_none_when_no_losses(self):
        bucket = _new_bucket()
        _update_bucket(bucket, simulate_range_outcome({'open': 101, 'high': 108, 'low': 99, 'close': 106}, RANGE_WIN))
        final = _finalize_bucket(bucket)
        # No stops ? profit_factor undefined
        self.assertIsNone(final['profit_factor'])

    def test_avg_win_pct_positive(self):
        bucket = _new_bucket()
        _update_bucket(bucket, simulate_range_outcome({'open': 101, 'high': 108, 'low': 99, 'close': 106}, RANGE_WIN))
        final = _finalize_bucket(bucket)
        self.assertGreater(final['avg_win_pct'], 0)

    def test_avg_loss_pct_positive_for_stops(self):
        bucket = _new_bucket()
        _update_bucket(bucket, simulate_range_outcome({'open': 101, 'high': 101, 'low': 97, 'close': 99}, RANGE_STOP))
        final = _finalize_bucket(bucket)
        # avg_loss_pct is stored as a positive number (absolute loss)
        self.assertGreater(final['avg_loss_pct'], 0)

    def test_max_consecutive_losses_tracked(self):
        bucket = _new_bucket()
        stop_bar = {'open': 101, 'high': 101, 'low': 97, 'close': 99}
        win_bar  = {'open': 101, 'high': 108, 'low': 99, 'close': 106}
        _update_bucket(bucket, simulate_range_outcome(stop_bar, RANGE_STOP))
        _update_bucket(bucket, simulate_range_outcome(stop_bar, RANGE_STOP))
        _update_bucket(bucket, simulate_range_outcome(stop_bar, RANGE_STOP))
        _update_bucket(bucket, simulate_range_outcome(win_bar, RANGE_WIN))
        final = _finalize_bucket(bucket)
        self.assertEqual(final['max_consecutive_losses'], 3)

    def test_max_consecutive_losses_resets_after_win(self):
        bucket = _new_bucket()
        stop_bar = {'open': 101, 'high': 101, 'low': 97, 'close': 99}
        win_bar  = {'open': 101, 'high': 108, 'low': 99, 'close': 106}
        # 2 losses, win, 1 loss ? max should be 2
        _update_bucket(bucket, simulate_range_outcome(stop_bar, RANGE_STOP))
        _update_bucket(bucket, simulate_range_outcome(stop_bar, RANGE_STOP))
        _update_bucket(bucket, simulate_range_outcome(win_bar, RANGE_WIN))
        _update_bucket(bucket, simulate_range_outcome(stop_bar, RANGE_STOP))
        final = _finalize_bucket(bucket)
        self.assertEqual(final['max_consecutive_losses'], 2)

    def test_internal_fields_not_in_output(self):
        final = _finalize_bucket(_new_bucket())
        for key in final:
            self.assertFalse(key.startswith('_'), f"Internal field leaked to output: {key}")


if __name__ == '__main__':
    unittest.main()
