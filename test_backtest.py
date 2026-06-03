"""Unit tests for backtest helpers"""
import unittest

from backtest import (
    MIN_ADEQUATE_SAMPLE,
    _finalize_bucket,
    _new_bucket,
    _stats,
    _update_bucket,
    simulate_range_outcome,
)


RANGE_WIN = {'buy_high': 100.0, 'sell_low': 107.0}
RANGE_STOP = {'buy_high': 100.0, 'sell_low': 110.0}
RANGE_MISS = {'buy_high': 100.0, 'sell_low': 110.0}
RANGE_NO_TRIGGER = {'buy_high': 95.0, 'sell_low': 105.0}


class SimulateRangeTests(unittest.TestCase):
    def test_win(self):
        bar = {'open': 101, 'high': 108, 'low': 99, 'close': 106}  # low=99 > stop(95)
        result = simulate_range_outcome(bar, RANGE_WIN)
        self.assertEqual(result['outcome'], 'win')
        self.assertTrue(result['triggered'])
        self.assertGreater(result['return_pct'], 0)

    def test_stop(self):
        # Default stop is buy_high * 0.95 = 95; low must dip to/below it.
        bar = {'open': 101, 'high': 101, 'low': 93, 'close': 96}
        result = simulate_range_outcome(bar, RANGE_STOP)
        self.assertEqual(result['outcome'], 'stop')
        self.assertEqual(result['return_pct'], -5.0)
        self.assertTrue(result['triggered'])

    def test_stop_respects_custom_pct(self):
        # With a 0.98 stop, a dip to 97 should stop out at -2%.
        bar = {'open': 101, 'high': 101, 'low': 97, 'close': 99}
        result = simulate_range_outcome(bar, RANGE_STOP, stop_loss_pct=0.98)
        self.assertEqual(result['outcome'], 'stop')
        self.assertEqual(result['return_pct'], -2.0)

    def test_no_stop_above_threshold(self):
        # low=97 is above the default 95 stop, so this is a miss, not a stop.
        bar = {'open': 101, 'high': 101, 'low': 97, 'close': 99}
        result = simulate_range_outcome(bar, RANGE_STOP)
        self.assertEqual(result['outcome'], 'miss')

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
        _update_bucket(bucket, simulate_range_outcome({'open': 101, 'high': 101, 'low': 93, 'close': 96}, RANGE_STOP))
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
        # Win returns 7%, stop returns -5%
        _update_bucket(bucket, simulate_range_outcome({'open': 101, 'high': 108, 'low': 99, 'close': 106}, RANGE_WIN))
        _update_bucket(bucket, simulate_range_outcome({'open': 101, 'high': 101, 'low': 93, 'close': 96}, RANGE_STOP))
        final = _finalize_bucket(bucket)
        # gross_profit = ~7%, gross_loss = 5%  => pf = ~1.4
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
        _update_bucket(bucket, simulate_range_outcome({'open': 101, 'high': 101, 'low': 93, 'close': 96}, RANGE_STOP))
        final = _finalize_bucket(bucket)
        # avg_loss_pct is stored as a positive number (absolute loss)
        self.assertGreater(final['avg_loss_pct'], 0)

    def test_max_consecutive_losses_tracked(self):
        bucket = _new_bucket()
        stop_bar = {'open': 101, 'high': 101, 'low': 93, 'close': 96}
        win_bar  = {'open': 101, 'high': 108, 'low': 99, 'close': 106}
        _update_bucket(bucket, simulate_range_outcome(stop_bar, RANGE_STOP))
        _update_bucket(bucket, simulate_range_outcome(stop_bar, RANGE_STOP))
        _update_bucket(bucket, simulate_range_outcome(stop_bar, RANGE_STOP))
        _update_bucket(bucket, simulate_range_outcome(win_bar, RANGE_WIN))
        final = _finalize_bucket(bucket)
        self.assertEqual(final['max_consecutive_losses'], 3)

    def test_max_consecutive_losses_resets_after_win(self):
        bucket = _new_bucket()
        stop_bar = {'open': 101, 'high': 101, 'low': 93, 'close': 96}
        win_bar  = {'open': 101, 'high': 108, 'low': 99, 'close': 106}
        # 2 losses, win, 1 loss ? max should be 2
        _update_bucket(bucket, simulate_range_outcome(stop_bar, RANGE_STOP))
        _update_bucket(bucket, simulate_range_outcome(stop_bar, RANGE_STOP))
        _update_bucket(bucket, simulate_range_outcome(win_bar, RANGE_WIN))
        _update_bucket(bucket, simulate_range_outcome(stop_bar, RANGE_STOP))
        final = _finalize_bucket(bucket)
        self.assertEqual(final['max_consecutive_losses'], 2)

    def test_negative_miss_counts_as_loss(self):
        # Triggered (low<=100), not stopped (low>95), not a win (high<110),
        # but closed below entry -> negative return must count toward losses.
        bucket = _new_bucket()
        bar = {'open': 99, 'high': 99, 'low': 96, 'close': 97}
        outcome = simulate_range_outcome(bar, RANGE_MISS)
        self.assertEqual(outcome['outcome'], 'miss')
        self.assertLess(outcome['return_pct'], 0)
        _update_bucket(bucket, outcome)
        final = _finalize_bucket(bucket)
        # avg_loss_pct must reflect this losing miss even though stops == 0
        self.assertEqual(final['stops'], 0)
        self.assertGreater(final['avg_loss_pct'], 0)
        self.assertEqual(final['profit_factor'], 0.0)  # loss present, zero profit

    def test_avg_loss_pct_not_inflated_by_stop_only_divisor(self):
        # One -5% stop and one -1% miss: avg loss must be 3%, not 6% (5+1)/1.
        bucket = _new_bucket()
        _update_bucket(bucket, simulate_range_outcome({'open': 101, 'high': 101, 'low': 93, 'close': 96}, RANGE_STOP))
        _update_bucket(bucket, simulate_range_outcome({'open': 99, 'high': 99, 'low': 96, 'close': 97}, RANGE_MISS))
        final = _finalize_bucket(bucket)
        self.assertAlmostEqual(final['avg_loss_pct'], 3.0, places=2)

    def test_internal_fields_not_in_output(self):
        final = _finalize_bucket(_new_bucket())
        for key in final:
            self.assertFalse(key.startswith('_'), f"Internal field leaked to output: {key}")


class StatsTests(unittest.TestCase):

    def test_empty_returns_zeroed(self):
        s = _stats([])
        self.assertEqual(s['n'], 0)
        self.assertFalse(s['significant'])
        self.assertFalse(s['sample_adequate'])
        self.assertIsNone(s['t_stat'])

    def test_single_value_no_std(self):
        s = _stats([1.5])
        self.assertEqual(s['n'], 1)
        self.assertAlmostEqual(s['mean'], 1.5)
        self.assertEqual(s['std'], 0.0)
        self.assertIsNone(s['t_stat'])  # cannot compute with n<2
        self.assertFalse(s['significant'])

    def test_sample_adequacy_threshold(self):
        self.assertFalse(_stats([0.1] * (MIN_ADEQUATE_SAMPLE - 1))['sample_adequate'])
        self.assertTrue(_stats([0.1] * MIN_ADEQUATE_SAMPLE)['sample_adequate'])

    def test_significant_requires_adequate_sample(self):
        # A tiny but perfectly consistent sample must NOT be called significant.
        small = _stats([1.0, 1.0, 1.0])
        self.assertFalse(small['significant'])
        self.assertFalse(small['sample_adequate'])

    def test_strong_consistent_signal_is_significant(self):
        # 40 trades, all clearly positive with low variance -> significant.
        rets = [1.0 + (0.01 if i % 2 else -0.01) for i in range(40)]
        s = _stats(rets)
        self.assertTrue(s['sample_adequate'])
        self.assertTrue(s['significant'])
        self.assertGreater(s['t_stat'], 1.96)

    def test_noisy_zero_mean_not_significant(self):
        # 40 trades centered on zero with high variance -> not significant.
        rets = [(-3.0 if i % 2 else 3.0) for i in range(40)]
        s = _stats(rets)
        self.assertTrue(s['sample_adequate'])
        self.assertFalse(s['significant'])

    def test_ci_brackets_mean(self):
        s = _stats([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertLessEqual(s['ci95_low'], s['mean'])
        self.assertGreaterEqual(s['ci95_high'], s['mean'])


class ExecutionModeTests(unittest.TestCase):
    BAR = {'open': 100.0, 'high': 104.0, 'low': 96.0, 'close': 102.0}
    PRED = {'buy_high': 98.0, 'sell_low': 110.0}  # entry 98, target unreached

    def test_market_hold_ignores_range_and_uses_open_close(self):
        out = simulate_range_outcome(self.BAR, self.PRED, execution='market_hold')
        self.assertTrue(out['triggered'])  # always "in" under market_hold
        # (close-open)/open = (102-100)/100 = 2%
        self.assertAlmostEqual(out['return_pct'], 2.0, places=2)

    def test_market_hold_triggers_even_when_limit_would_not(self):
        # low (96) > buy_high (95) -> limit modes don't trigger, market_hold does
        bar = {'open': 100.0, 'high': 104.0, 'low': 96.0, 'close': 101.0}
        pred = {'buy_high': 95.0, 'sell_low': 110.0}
        self.assertFalse(simulate_range_outcome(bar, pred, execution='limit_stop')['triggered'])
        self.assertTrue(simulate_range_outcome(bar, pred, execution='market_hold')['triggered'])

    def test_limit_hold_exits_at_close_no_stop(self):
        # Dips to entry 98 (low 96 <= 98) but no stop; exit at close 102.
        out = simulate_range_outcome(self.BAR, self.PRED, execution='limit_hold')
        self.assertTrue(out['triggered'])
        # (102-98)/98 ~ +4.08%
        self.assertAlmostEqual(out['return_pct'], (102 - 98) / 98 * 100, places=2)

    def test_limit_hold_no_trigger_when_price_stays_above_buy_high(self):
        bar = {'open': 100.0, 'high': 104.0, 'low': 99.0, 'close': 103.0}
        pred = {'buy_high': 98.0, 'sell_low': 110.0}
        out = simulate_range_outcome(bar, pred, execution='limit_hold')
        self.assertFalse(out['triggered'])

    def test_default_execution_unchanged(self):
        # Default path must still behave as the original limit_stop logic.
        out = simulate_range_outcome({'open': 101, 'high': 108, 'low': 99, 'close': 106}, RANGE_WIN)
        self.assertEqual(out['outcome'], 'win')


class SignificanceWiringTests(unittest.TestCase):

    def test_finalize_bucket_includes_significance(self):
        bucket = _new_bucket()
        _update_bucket(bucket, simulate_range_outcome({'open': 101, 'high': 108, 'low': 99, 'close': 106}, RANGE_WIN))
        final = _finalize_bucket(bucket)
        self.assertIn('significance', final)
        self.assertEqual(final['significance']['n'], 1)

    def test_returns_list_excluded_from_output(self):
        final = _finalize_bucket(_new_bucket())
        self.assertNotIn('_returns', final)


if __name__ == '__main__':
    unittest.main()
