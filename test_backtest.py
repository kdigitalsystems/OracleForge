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


if __name__ == '__main__':
    unittest.main()
