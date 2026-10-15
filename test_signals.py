"""Unit tests for signals.py"""
import unittest

from signals import (
    build_enriched_predictions,
    cap_buy_at_close,
    classify_opportunity,
    extract_model_predictions,
    parse_ticker_list,
    weighted_consensus_ranges,
)

RANGE_A = {'buy_low': 95.0, 'buy_high': 97.0, 'sell_low': 105.0, 'sell_high': 108.0, 'rationale': ''}
RANGE_B = {'buy_low': 94.0, 'buy_high': 96.0, 'sell_low': 104.0, 'sell_high': 107.0, 'rationale': ''}


class ExtractTests(unittest.TestCase):
    def test_extract_new_format(self):
        entry = {
            'close': 100.0,
            'signal': 'ACTIVE',
            'models': {
                'model_a': RANGE_A,
                'model_b': RANGE_B,
            },
        }
        result = extract_model_predictions(entry)
        self.assertIn('model_a', result)
        self.assertEqual(result['model_a']['buy_low'], 95.0)

    def test_extract_missing_models(self):
        self.assertEqual(extract_model_predictions({}), {})
        self.assertEqual(extract_model_predictions({'close': 100.0}), {})

    def test_extract_ignores_non_range_models(self):
        entry = {
            'models': {
                'good_model': RANGE_A,
                'bad_model': 200.0,  # old HOD float format
            }
        }
        result = extract_model_predictions(entry)
        self.assertIn('good_model', result)
        self.assertNotIn('bad_model', result)


class ConsensusTests(unittest.TestCase):
    def test_equal_weights(self):
        preds = {'a': RANGE_A, 'b': RANGE_B}
        scores = {'a': 5.0, 'b': 5.0}
        result = weighted_consensus_ranges(preds, scores)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result['buy_low'], 94.5, places=1)
        self.assertAlmostEqual(result['buy_high'], 96.5, places=1)

    def test_skewed_weights(self):
        preds = {'a': RANGE_A, 'b': RANGE_B}
        scores = {'a': 10.0, 'b': 0.0}
        result = weighted_consensus_ranges(preds, scores)
        # model_b gets minimum weight 0.1, result should be very close to model_a
        self.assertAlmostEqual(result['buy_low'], 95.0, delta=0.2)

    def test_empty(self):
        self.assertIsNone(weighted_consensus_ranges({}, {}))

    def test_only_one_valid_model_returns_none(self):
        # RANGE_A is valid; 'bad' has incoherent range (buy_high > sell_low)
        bad_range = {'buy_low': 95.0, 'buy_high': 110.0, 'sell_low': 105.0, 'sell_high': 108.0}
        preds = {'a': RANGE_A, 'bad': bad_range}
        scores = {'a': 5.0, 'bad': 5.0}
        result = weighted_consensus_ranges(preds, scores, min_agreeing_models=2)
        self.assertIsNone(result)  # only 1 valid model ? below threshold

    def test_skipped_model_not_counted(self):
        # A model that returned a 'skipped' dict should not count toward valid models
        skipped = {'skipped': True, 'reason': 'upcoming_earnings'}
        preds = {'a': RANGE_A, 'skipped_model': skipped}
        scores = {'a': 5.0, 'skipped_model': 5.0}
        result = weighted_consensus_ranges(preds, scores, min_agreeing_models=2)
        self.assertIsNone(result)

    def test_fallback_excluded_from_consensus(self):
        # A fallback-tagged prediction should not count toward the consensus
        fallback_range = {**RANGE_A, 'fallback': True}
        preds = {'real_model': RANGE_A, 'fallback_model': fallback_range}
        scores = {'real_model': 5.0, 'fallback_model': 5.0}
        # Only 1 real model ? should be None with min_agreeing_models=2
        result = weighted_consensus_ranges(preds, scores, min_agreeing_models=2)
        self.assertIsNone(result, "Fallback predictions must not count toward min_agreeing_models")

    def test_fallback_does_not_skew_consensus(self):
        # Even with min=1, a fallback pred should not pull the consensus toward its range
        real_range = {'buy_low': 95.0, 'buy_high': 97.0, 'sell_low': 105.0, 'sell_high': 108.0}
        # Fallback range has wildly different values
        fallback_range = {'buy_low': 50.0, 'buy_high': 55.0, 'sell_low': 80.0, 'sell_high': 85.0,
                          'fallback': True}
        preds = {'real': real_range, 'fb': fallback_range}
        scores = {'real': 5.0, 'fb': 5.0}
        result = weighted_consensus_ranges(preds, scores, min_agreeing_models=1)
        self.assertIsNotNone(result)
        # buy_high must match the real range, not be pulled down by fallback
        self.assertAlmostEqual(result['buy_high'], 97.0, delta=0.1)

    def test_cv_gate_rejects_high_disagreement(self):
        # Two models with very different buy_high values ? high CV ? should be rejected
        range_low  = {'buy_low': 90.0, 'buy_high': 92.0, 'sell_low': 100.0, 'sell_high': 105.0}
        range_high = {'buy_low': 130.0, 'buy_high': 135.0, 'sell_low': 150.0, 'sell_high': 155.0}
        preds = {'a': range_low, 'b': range_high}
        scores = {'a': 5.0, 'b': 5.0}
        # max_cv=0.10 should reject this extreme disagreement
        result = weighted_consensus_ranges(preds, scores, max_cv=0.10)
        self.assertIsNone(result, "CV gate should reject models with >10%% disagreement")

    def test_cv_gate_passes_close_agreement(self):
        # Two models with nearly identical ranges ? low CV ? should pass
        range_x = {'buy_low': 95.0, 'buy_high': 97.0, 'sell_low': 105.0, 'sell_high': 108.0}
        range_y = {'buy_low': 95.5, 'buy_high': 97.5, 'sell_low': 105.5, 'sell_high': 108.5}
        preds = {'a': range_x, 'b': range_y}
        scores = {'a': 5.0, 'b': 5.0}
        result = weighted_consensus_ranges(preds, scores, max_cv=0.10)
        self.assertIsNotNone(result, "CV gate should pass models that agree within 10%%")

    def test_cv_gate_none_disables_check(self):
        # max_cv=None means no CV gate ? high-disagreement models should still produce consensus
        range_low  = {'buy_low': 90.0, 'buy_high': 92.0, 'sell_low': 100.0, 'sell_high': 105.0}
        range_high = {'buy_low': 130.0, 'buy_high': 135.0, 'sell_low': 150.0, 'sell_high': 155.0}
        preds = {'a': range_low, 'b': range_high}
        scores = {'a': 5.0, 'b': 5.0}
        result = weighted_consensus_ranges(preds, scores, max_cv=None)
        self.assertIsNotNone(result, "max_cv=None should disable the CV gate entirely")

    def test_consensus_cv_is_exposed(self):
        # The disagreement (CV) must be surfaced on the result for the dashboard.
        result = weighted_consensus_ranges({'a': RANGE_A, 'b': RANGE_B}, {'a': 5.0, 'b': 5.0})
        self.assertIn('consensus_cv', result)
        self.assertIsInstance(result['consensus_cv'], float)
        self.assertGreaterEqual(result['consensus_cv'], 0.0)

    def test_consensus_cv_larger_when_models_disagree_more(self):
        close = {'a': RANGE_A, 'b': RANGE_B}            # nearly identical -> small CV
        far = {
            'a': {'buy_low': 90.0, 'buy_high': 92.0, 'sell_low': 100.0, 'sell_high': 105.0},
            'b': {'buy_low': 108.0, 'buy_high': 110.0, 'sell_low': 120.0, 'sell_high': 125.0},
        }
        cv_close = weighted_consensus_ranges(close, {'a': 5.0, 'b': 5.0})['consensus_cv']
        cv_far = weighted_consensus_ranges(far, {'a': 5.0, 'b': 5.0})['consensus_cv']
        self.assertGreater(cv_far, cv_close)


class ClassifyTests(unittest.TestCase):
    def test_active_signal(self):
        consensus = {'buy_low': 95.0, 'buy_high': 97.0, 'sell_low': 105.0, 'sell_high': 108.0}
        result = classify_opportunity(close=100.0, consensus=consensus)
        self.assertEqual(result['signal'], 'ACTIVE')
        self.assertGreater(result['upside_pct'], 0)

    def test_stale_signal(self):
        consensus = {'buy_low': 95.0, 'buy_high': 97.0, 'sell_low': 99.0, 'sell_high': 101.0}
        result = classify_opportunity(close=102.0, consensus=consensus)
        self.assertEqual(result['signal'], 'STALE')

    def test_stale_upside_is_none(self):
        # upside_pct should be None for STALE signals — the opportunity is already missed
        consensus = {'buy_low': 95.0, 'buy_high': 97.0, 'sell_low': 99.0, 'sell_high': 101.0}
        result = classify_opportunity(close=102.0, consensus=consensus)
        self.assertIsNone(result['upside_pct'], 'STALE upside_pct should be None')

    def test_skip_low_upside(self):
        consensus = {'buy_low': 99.0, 'buy_high': 99.5, 'sell_low': 100.1, 'sell_high': 100.5}
        result = classify_opportunity(
            close=100.0,
            consensus=consensus,
            config={'min_upside_pct': 1.0},
        )
        self.assertEqual(result['signal'], 'SKIP')

    def test_no_consensus(self):
        result = classify_opportunity(close=100.0, consensus=None)
        self.assertEqual(result['signal'], 'SKIP')

    def test_buy_high_equals_sell_low_is_skip(self):
        # Degenerate range where entry == target
        consensus = {'buy_low': 95.0, 'buy_high': 100.0, 'sell_low': 100.0, 'sell_high': 102.0}
        result = classify_opportunity(close=98.0, consensus=consensus)
        self.assertEqual(result['signal'], 'SKIP')

    def test_upside_pct_measured_from_entry_not_close(self):
        # close=90, buy_high=98, sell_low=100
        # upside from entry = (100-98)/98 = 2.04%
        consensus = {'buy_low': 95.0, 'buy_high': 98.0, 'sell_low': 100.0, 'sell_high': 102.0}
        result = classify_opportunity(
            close=90.0, consensus=consensus, config={'min_upside_pct': 1.0}
        )
        self.assertIsNotNone(result['upside_pct'])
        self.assertLess(result['upside_pct'], 5.0)

    def test_wide_spread_is_skipped(self):
        # buy range is 10% wide ? should be rejected by max_spread_pct=3
        consensus = {'buy_low': 90.0, 'buy_high': 100.0, 'sell_low': 110.0, 'sell_high': 115.0}
        result = classify_opportunity(
            close=100.0, consensus=consensus,
            config={'min_upside_pct': 1.0, 'max_spread_pct': 3.0},
        )
        self.assertEqual(result['signal'], 'SKIP')


class CapBuyAtCloseTests(unittest.TestCase):
    ABOVE = {'buy_low': 99.0, 'buy_high': 101.0, 'sell_low': 105.0, 'sell_high': 108.0}

    def test_buy_high_above_close_is_capped_and_keeps_the_model_level(self):
        out = cap_buy_at_close(self.ABOVE, 100.0)
        self.assertEqual((out['buy_low'], out['buy_high'], out['buy_high_model']), (99.0, 100.0, 101.0))
        self.assertEqual((out['sell_low'], out['sell_high']), (105.0, 108.0))

    def test_buy_low_is_pulled_down_when_the_whole_range_is_above_the_close(self):
        out = cap_buy_at_close({**self.ABOVE, 'buy_low': 100.5}, 100.0)
        self.assertEqual((out['buy_low'], out['buy_high']), (100.0, 100.0))

    def test_unchanged_when_buy_high_is_at_or_below_the_close(self):
        for close in (101.0, 102.5):
            self.assertIs(cap_buy_at_close(self.ABOVE, close), self.ABOVE)

    def test_cap_is_never_above_the_close_and_is_idempotent(self):
        for close in (11.956, 11.95, 11.954, 0.37):
            out = cap_buy_at_close({'buy_low': 0.3, 'buy_high': 99.0, 'sell_low': 120.0, 'sell_high': 125.0}, close)
            self.assertLessEqual(out['buy_high'], close)
            self.assertGreater(out['buy_high'], close - 0.0100001)
            self.assertEqual(cap_buy_at_close(out, close), out)

    def test_input_is_not_mutated(self):
        original = dict(self.ABOVE)
        cap_buy_at_close(self.ABOVE, 100.0)
        self.assertEqual(self.ABOVE, original)


class ClassifyCapTests(unittest.TestCase):
    ABOVE = {'buy_low': 99.0, 'buy_high': 101.0, 'sell_low': 105.0, 'sell_high': 108.0}

    def test_active_signal_orders_at_the_close_not_above_it(self):
        result = classify_opportunity(100.0, self.ABOVE, {'min_upside_pct': 1.5})
        self.assertEqual(result['signal'], 'ACTIVE')
        self.assertEqual(result['consensus']['buy_high'], 100.0)
        self.assertEqual(result['consensus']['buy_high_model'], 101.0)

    def test_membership_and_upside_still_come_from_the_models_buy_high(self):
        # (105 - 101) / 101 = 3.96%; the cap must not change either the
        # ACTIVE decision or the reported upside.
        result = classify_opportunity(100.0, self.ABOVE, {'min_upside_pct': 1.5})
        self.assertEqual(result['upside_pct'], 3.96)
        thin = {'buy_low': 99.0, 'buy_high': 101.0, 'sell_low': 102.0, 'sell_high': 103.0}
        self.assertEqual(classify_opportunity(100.0, thin, {'min_upside_pct': 1.5})['signal'], 'SKIP')

    def test_buy_high_below_close_is_left_alone(self):
        consensus = {'buy_low': 95.0, 'buy_high': 97.0, 'sell_low': 105.0, 'sell_high': 108.0}
        result = classify_opportunity(100.0, consensus, {'min_upside_pct': 1.5})
        self.assertEqual(result['consensus'], consensus)
        self.assertNotIn('buy_high_model', result['consensus'])

    def test_switch_off_leaves_the_consensus_alone(self):
        result = classify_opportunity(100.0, self.ABOVE, {'min_upside_pct': 1.5, 'cap_buy_at_close': False})
        self.assertEqual(result['signal'], 'ACTIVE')
        self.assertEqual(result['consensus'], self.ABOVE)

    def test_only_active_rows_are_capped(self):
        stale = {'buy_low': 95.0, 'buy_high': 97.0, 'sell_low': 99.0, 'sell_high': 101.0}
        self.assertEqual(classify_opportunity(102.0, stale)['consensus'], stale)
        thin = {'buy_low': 99.0, 'buy_high': 101.0, 'sell_low': 102.0, 'sell_high': 103.0}
        self.assertEqual(classify_opportunity(100.0, thin, {'min_upside_pct': 1.5})['consensus'], thin)

    def test_wide_spread_is_still_judged_on_the_models_range(self):
        wide = {'buy_low': 95.0, 'buy_high': 101.0, 'sell_low': 110.0, 'sell_high': 115.0}
        result = classify_opportunity(100.0, wide, {'min_upside_pct': 1.0, 'max_spread_pct': 3.0})
        self.assertEqual(result['signal'], 'SKIP')

    def test_enriched_predictions_store_the_capped_consensus(self):
        raw = {'AAA': {'m1': {**self.ABOVE, 'rationale': ''}, 'm2': {**self.ABOVE, 'rationale': ''}}}
        out = build_enriched_predictions(raw, {'AAA': 100.0}, {'m1': 5.0, 'm2': 5.0},
                                         {'min_upside_pct': 1.5, 'min_agreeing_models': 2})
        self.assertEqual(out['AAA']['signal'], 'ACTIVE')
        self.assertEqual(out['AAA']['consensus']['buy_high'], 100.0)
        self.assertEqual(out['AAA']['consensus']['buy_high_model'], 101.0)


class BuildEnrichedTests(unittest.TestCase):
    def test_build_enriched(self):
        raw = {'NVDA': {'model_a': RANGE_A, 'model_b': RANGE_B}}
        closes = {'NVDA': 100.0}
        scores = {'model_a': 8.0, 'model_b': 4.0}
        enriched = build_enriched_predictions(raw, closes, scores)
        self.assertIn('NVDA', enriched)
        entry = enriched['NVDA']
        self.assertEqual(entry['close'], 100.0)
        self.assertIn('signal', entry)
        self.assertIn('consensus', entry)
        self.assertIn('models', entry)

    def test_missing_close_skipped(self):
        raw = {'AAPL': {'model_a': RANGE_A}}
        enriched = build_enriched_predictions(raw, {}, {})
        self.assertNotIn('AAPL', enriched)

    def test_build_enriched_cv_gate_via_config(self):
        # Config with max_consensus_cv should be passed through to weighted_consensus_ranges
        range_low  = {'buy_low': 90.0, 'buy_high': 92.0, 'sell_low': 100.0, 'sell_high': 105.0}
        range_high = {'buy_low': 130.0, 'buy_high': 135.0, 'sell_low': 150.0, 'sell_high': 155.0}
        raw = {'TSLA': {'a': range_low, 'b': range_high}}
        closes = {'TSLA': 100.0}
        scores = {'a': 5.0, 'b': 5.0}
        config = {'min_agreeing_models': 2, 'max_consensus_cv': 0.10, 'min_upside_pct': 1.0}
        enriched = build_enriched_predictions(raw, closes, scores, config)
        # CV is too high ? consensus should be None, signal should be SKIP
        self.assertIn('TSLA', enriched)
        self.assertIsNone(enriched['TSLA']['consensus'])
        self.assertEqual(enriched['TSLA']['signal'], 'SKIP')


class UtilTests(unittest.TestCase):
    def test_parse_ticker_list(self):
        self.assertEqual(parse_ticker_list('nvda, aapl'), ['NVDA', 'AAPL'])
        self.assertIsNone(parse_ticker_list(None))
        self.assertIsNone(parse_ticker_list(''))


if __name__ == '__main__':
    unittest.main()
