"""Unit tests for signals.py"""
import unittest

from signals import (
    build_enriched_predictions,
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
        self.assertIsNone(result)  # only 1 valid model — below threshold

    def test_skipped_model_not_counted(self):
        # A model that returned a 'skipped' dict should not count toward valid models
        skipped = {'skipped': True, 'reason': 'upcoming_earnings'}
        preds = {'a': RANGE_A, 'skipped_model': skipped}
        scores = {'a': 5.0, 'skipped_model': 5.0}
        result = weighted_consensus_ranges(preds, scores, min_agreeing_models=2)
        self.assertIsNone(result)


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
        # upside from close = (100-90)/90 = 11.1%  ← old (wrong)
        # upside from entry = (100-98)/98 = 2.04%  ← new (correct)
        consensus = {'buy_low': 95.0, 'buy_high': 98.0, 'sell_low': 100.0, 'sell_high': 102.0}
        result = classify_opportunity(
            close=90.0, consensus=consensus, config={'min_upside_pct': 1.0}
        )
        self.assertIsNotNone(result['upside_pct'])
        # Should be ~2%, not ~11%
        self.assertLess(result['upside_pct'], 5.0)

    def test_wide_spread_is_skipped(self):
        # buy range is 10% wide — should be rejected by max_spread_pct=3
        consensus = {'buy_low': 90.0, 'buy_high': 100.0, 'sell_low': 110.0, 'sell_high': 115.0}
        result = classify_opportunity(
            close=100.0, consensus=consensus,
            config={'min_upside_pct': 1.0, 'max_spread_pct': 3.0},
        )
        self.assertEqual(result['signal'], 'SKIP')


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


class UtilTests(unittest.TestCase):
    def test_parse_ticker_list(self):
        self.assertEqual(parse_ticker_list('nvda, aapl'), ['NVDA', 'AAPL'])
        self.assertIsNone(parse_ticker_list(None))
        self.assertIsNone(parse_ticker_list(''))


if __name__ == '__main__':
    unittest.main()
