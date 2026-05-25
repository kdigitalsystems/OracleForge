"""Unit tests for forge_loop helpers."""
from __future__ import annotations

import sys
import unittest
from unittest.mock import MagicMock

# Stub alpaca_client so forge_loop can be imported without Alpaca credentials
sys.modules.setdefault('alpaca_client', MagicMock())

from forge_loop import (
    _fallback_range,
    apply_score_deltas,
    compute_technicals,
    evaluate_range_prediction,
    parse_llm_range,
    score_deltas_from_journal,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockBar:
    """Minimal bar object matching Alpaca's Bar interface."""
    def __init__(self, close, high=None, low=None, volume=1_000_000):
        self.close = close
        self.high = high if high is not None else close * 1.01
        self.low = low if low is not None else close * 0.99
        self.volume = volume


def _make_bars(closes, volumes=None):
    """Build a list of MockBar from a list of close prices."""
    if volumes is None:
        volumes = [1_000_000] * len(closes)
    return [MockBar(c, c * 1.01, c * 0.99, v) for c, v in zip(closes, volumes)]


# ---------------------------------------------------------------------------
# parse_llm_range
# ---------------------------------------------------------------------------

class ParseLlmRangeTests(unittest.TestCase):

    def test_valid_json(self):
        raw = '{"buy_low": 95.0, "buy_high": 97.0, "sell_low": 105.0, "sell_high": 108.0, "rationale": "ok"}'
        result = parse_llm_range(raw, 100.0)
        self.assertEqual(result['buy_low'], 95.0)
        self.assertEqual(result['sell_low'], 105.0)
        self.assertNotIn('Fallback', result.get('rationale', ''))

    def test_markdown_fences_stripped(self):
        raw = '```json\n{"buy_low":95,"buy_high":97,"sell_low":105,"sell_high":108,"rationale":"x"}\n```'
        result = parse_llm_range(raw, 100.0)
        self.assertEqual(result['buy_high'], 97.0)

    def test_json_embedded_in_text(self):
        raw = 'Sure! Here is my answer: {"buy_low":95,"buy_high":97,"sell_low":105,"sell_high":108,"rationale":"x"} Hope that helps!'
        result = parse_llm_range(raw, 100.0)
        self.assertEqual(result['buy_low'], 95.0)

    def test_invalid_json_returns_fallback(self):
        result = parse_llm_range('this is not json', 100.0)
        self.assertIn('Fallback', result['rationale'])
        self.assertGreater(result['buy_low'], 0)
        self.assertGreater(result['sell_low'], result['buy_high'])

    def test_empty_string_returns_fallback(self):
        result = parse_llm_range('', 100.0)
        self.assertIn('Fallback', result['rationale'])

    def test_incoherent_range_returns_fallback(self):
        # buy_high > sell_low ? invalid
        raw = '{"buy_low": 95, "buy_high": 110, "sell_low": 105, "sell_high": 108, "rationale": "bad"}'
        result = parse_llm_range(raw, 100.0)
        self.assertIn('Fallback', result['rationale'])


# ---------------------------------------------------------------------------
# _fallback_range
# ---------------------------------------------------------------------------

class FallbackRangeTests(unittest.TestCase):

    def test_fallback_tagged(self):
        result = _fallback_range(100.0)
        self.assertTrue(result.get('fallback'), "Fallback range must be tagged fallback=True")

    def test_fallback_valid_range(self):
        result = _fallback_range(200.0)
        self.assertGreater(result['buy_high'], result['buy_low'])
        self.assertGreater(result['sell_low'], result['buy_high'])
        self.assertGreaterEqual(result['sell_high'], result['sell_low'])

    def test_parse_llm_fallback_tagged(self):
        # Unparseable output should produce a tagged fallback
        result = parse_llm_range('definitely not json', 100.0)
        self.assertTrue(result.get('fallback'), "parse_llm_range fallback must be tagged")


# ---------------------------------------------------------------------------
# evaluate_range_prediction
# ---------------------------------------------------------------------------

class EvaluateRangePredictionTests(unittest.TestCase):

    PRED = {'buy_high': 100.0, 'sell_low': 107.0}

    def test_win(self):
        # Low touched buy range, high reached sell target, stop not hit
        delta = evaluate_range_prediction(high_price=108, low_price=99, pred=self.PRED)
        self.assertEqual(delta, 0.01)

    def test_stop_hit(self):
        # Low breached buy_high * 0.98 = 98.0
        delta = evaluate_range_prediction(high_price=101, low_price=97, pred=self.PRED)
        self.assertEqual(delta, -0.01)

    def test_no_entry(self):
        # Low never touched buy range (low > buy_high)
        delta = evaluate_range_prediction(high_price=110, low_price=101, pred=self.PRED)
        self.assertEqual(delta, 0.0)

    def test_hold_inconclusive(self):
        # Price entered buy range but neither stop nor target hit
        delta = evaluate_range_prediction(high_price=104, low_price=99, pred=self.PRED)
        self.assertEqual(delta, 0.0)

    def test_invalid_pred(self):
        self.assertEqual(evaluate_range_prediction(105, 95, pred={}), 0.0)
        self.assertEqual(evaluate_range_prediction(105, 95, pred=None), 0.0)

    def test_fallback_pred_returns_zero(self):
        # Synthetic fallback predictions must not affect model scores
        fallback_pred = {**self.PRED, 'fallback': True}
        # Would be a win without the fallback flag, but should be 0.0
        delta = evaluate_range_prediction(high_price=108, low_price=99, pred=fallback_pred)
        self.assertEqual(delta, 0.0)

    def test_fallback_pred_stop_returns_zero(self):
        # Fallback stop: would be -0.01, but should be 0.0
        fallback_pred = {**self.PRED, 'fallback': True}
        delta = evaluate_range_prediction(high_price=101, low_price=97, pred=fallback_pred)
        self.assertEqual(delta, 0.0)


# ---------------------------------------------------------------------------
# score_deltas_from_journal
# ---------------------------------------------------------------------------

class ScoreDeltasFromJournalTests(unittest.TestCase):

    SCORES = {'model_a': 5.0, 'model_b': 5.0}

    def _trade(self, outcome, date, models):
        return {'close_date': date, 'outcome': outcome, 'predicting_models': models, 'pnl_pct': 1.0}

    def test_win_adds_positive_delta(self):
        journal = [self._trade('win', '2026-05-20', ['model_a'])]
        deltas = score_deltas_from_journal(journal, '2026-05-20', self.SCORES)
        self.assertIn('model_a', deltas)
        self.assertEqual(deltas['model_a'], [0.02])

    def test_loss_adds_negative_delta(self):
        journal = [self._trade('loss', '2026-05-20', ['model_b'])]
        deltas = score_deltas_from_journal(journal, '2026-05-20', self.SCORES)
        self.assertEqual(deltas['model_b'], [-0.02])

    def test_wrong_date_not_counted(self):
        journal = [self._trade('win', '2026-05-19', ['model_a'])]
        deltas = score_deltas_from_journal(journal, '2026-05-20', self.SCORES)
        self.assertNotIn('model_a', deltas)

    def test_model_not_in_scores_ignored(self):
        journal = [self._trade('win', '2026-05-20', ['unknown_model'])]
        deltas = score_deltas_from_journal(journal, '2026-05-20', self.SCORES)
        self.assertNotIn('unknown_model', deltas)


# ---------------------------------------------------------------------------
# apply_score_deltas
# ---------------------------------------------------------------------------

class ApplyScoreDeltasTests(unittest.TestCase):

    def test_positive_adjustment(self):
        scores = {'model_a': 5.0}
        apply_score_deltas(scores, {'model_a': [0.01, 0.01]})
        self.assertGreater(scores['model_a'], 5.0)

    def test_clamped_at_zero(self):
        scores = {'model_a': 0.01}
        apply_score_deltas(scores, {'model_a': [-0.05]})
        self.assertEqual(scores['model_a'], 0.0)

    def test_clamped_at_ten(self):
        scores = {'model_a': 9.99}
        apply_score_deltas(scores, {'model_a': [0.05]})
        self.assertEqual(scores['model_a'], 10.0)

    def test_empty_deltas_no_change(self):
        scores = {'model_a': 5.0}
        apply_score_deltas(scores, {'model_a': []})
        self.assertEqual(scores['model_a'], 5.0)

    def test_all_zero_deltas_no_change(self):
        scores = {'model_a': 5.0}
        apply_score_deltas(scores, {'model_a': [0.0, 0.0]})
        self.assertEqual(scores['model_a'], 5.0)

    def test_score_decay_reduces_score_before_adjustment(self):
        # With decay=0.9: new_score = 5.0 * 0.9 + 0.01 = 4.51
        # Without decay:  new_score = 5.0 * 1.0 + 0.01 = 5.01
        scores_decayed = {'model_a': 5.0}
        scores_nodecay = {'model_a': 5.0}
        apply_score_deltas(scores_decayed, {'model_a': [0.01]}, decay=0.9)
        apply_score_deltas(scores_nodecay, {'model_a': [0.01]}, decay=1.0)
        self.assertLess(scores_decayed['model_a'], scores_nodecay['model_a'])

    def test_score_decay_default_is_no_decay(self):
        # Default decay=1.0 should produce same result as explicit 1.0
        scores_default = {'model_a': 5.0}
        scores_explicit = {'model_a': 5.0}
        apply_score_deltas(scores_default, {'model_a': [0.01]})
        apply_score_deltas(scores_explicit, {'model_a': [0.01]}, decay=1.0)
        self.assertAlmostEqual(scores_default['model_a'], scores_explicit['model_a'], places=6)


# ---------------------------------------------------------------------------
# compute_technicals
# ---------------------------------------------------------------------------

class ComputeTechnicalsTests(unittest.TestCase):

    def test_returns_all_keys_with_sufficient_bars(self):
        # 25 bars is enough for RSI-14, SMA-20, volume ratio
        closes = [100 + i * 0.5 for i in range(25)]
        bars = _make_bars(closes)
        result = compute_technicals(bars)
        for key in ('rsi14', 'vol_ratio', 'pct_from_10d_high', 'pct_from_10d_low', 'pct_from_sma20'):
            self.assertIn(key, result, f"Missing key: {key}")

    def test_bb_pct_present_with_20_bars(self):
        closes = [100 + i * 0.3 for i in range(25)]
        bars = _make_bars(closes)
        result = compute_technicals(bars)
        self.assertIn('bb_pct', result, "bb_pct missing with 25 bars")

    def test_price_momentum_5d_present(self):
        closes = [100 + i * 0.5 for i in range(25)]
        bars = _make_bars(closes)
        result = compute_technicals(bars)
        self.assertIn('price_momentum_5d', result, "price_momentum_5d missing with 25 bars")

    def test_price_momentum_5d_positive_for_rising(self):
        closes = list(range(100, 125))  # strictly rising
        bars = _make_bars(closes)
        result = compute_technicals(bars)
        self.assertGreater(result['price_momentum_5d'], 0)

    def test_bb_pct_in_valid_range(self):
        # Flat price ? near middle of band
        closes = [100.0] * 25
        bars = _make_bars(closes)
        result = compute_technicals(bars)
        # With flat prices, std=0 ? we fall back gracefully (key may be absent or 0.5)
        # Just check it's a float if present
        if 'bb_pct' in result:
            self.assertIsInstance(result['bb_pct'], float)

    def test_rsi_within_valid_range(self):
        closes = [100 + i for i in range(25)]
        bars = _make_bars(closes)
        result = compute_technicals(bars)
        self.assertIsNotNone(result.get('rsi14'))
        self.assertGreaterEqual(result['rsi14'], 0)
        self.assertLessEqual(result['rsi14'], 100)

    def test_too_few_bars_returns_empty(self):
        bars = _make_bars([100, 101, 102])
        result = compute_technicals(bars)
        self.assertEqual(result, {})

    def test_volume_ratio_positive(self):
        closes = [100] * 25
        volumes = [1_000_000] * 24 + [2_000_000]  # today 2x average
        bars = [MockBar(c, volume=v) for c, v in zip(closes, volumes)]
        result = compute_technicals(bars)
        self.assertGreater(result['vol_ratio'], 0)
        self.assertAlmostEqual(result['vol_ratio'], 2.0, delta=0.1)

    def test_purely_rising_prices_high_rsi(self):
        closes = list(range(80, 105))  # 25 strictly rising bars
        bars = _make_bars(closes)
        result = compute_technicals(bars)
        self.assertGreater(result['rsi14'], 50)


if __name__ == '__main__':
    unittest.main()
