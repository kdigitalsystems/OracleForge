"""Unit tests for forge_loop helpers."""
from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta, timezone
import unittest
from unittest.mock import MagicMock, patch

from forge_loop import (
    _fallback_range,
    _trade_score_delta,
    _with_timeout,
    apply_score_deltas,
    bar_is_after_prior,
    compute_technicals,
    evaluate_range_prediction,
    fetch_all_bars,
    fetch_recent_news,
    format_recent_ohlc,
    parse_llm_range,
    previous_prediction_date,
    score_deltas_from_journal,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockBar:
    """Minimal bar object matching Alpaca's Bar interface."""
    def __init__(self, close, high=None, low=None, volume=1_000_000,
                 open=None, timestamp=None):
        self.close = close
        self.high = high if high is not None else close * 1.01
        self.low = low if low is not None else close * 0.99
        self.open = open if open is not None else close
        self.volume = volume
        self.timestamp = timestamp


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

    def test_inline_think_block_with_echoed_template_is_skipped(self):
        # Older Ollama inlines deepseek-r1 reasoning in the response. The
        # template echo inside it used to be the only {...} tried.
        raw = ('<think>The format is {"buy_low": <number>, "buy_high": <number>}. '
               'Support is near 95.</think>\n'
               '{"buy_low":95,"buy_high":97,"sell_low":105,"sell_high":108,"rationale":"x"}')
        result = parse_llm_range(raw, 100.0)
        self.assertFalse(result.get('fallback'))
        self.assertEqual(result['buy_high'], 97.0)

    def test_draft_answer_in_reasoning_is_not_used(self):
        raw = ('<think>Draft: {"buy_low":90,"buy_high":91,"sell_low":120,"sell_high":121,"rationale":"d"} '
               'too wide, tighten it.</think>'
               '{"buy_low":95,"buy_high":97,"sell_low":105,"sell_high":108,"rationale":"final"}')
        result = parse_llm_range(raw, 100.0)
        self.assertEqual((result['buy_low'], result['rationale']), (95.0, 'final'))

    def test_only_closing_think_tag(self):
        raw = ('Reasoning about {"buy_low": <number>} here...</think>\n\n'
               '```json\n{"buy_low":95,"buy_high":97,"sell_low":105,"sell_high":108,"rationale":"x"}\n```')
        self.assertEqual(parse_llm_range(raw, 100.0)['sell_low'], 105.0)

    def test_unclosed_think_block_is_a_fallback(self):
        # Reasoning cut off before an answer: a draft inside must not be used.
        raw = '<think>Maybe {"buy_low":90,"buy_high":91,"sell_low":120,"sell_high":121,"rationale":"d"}'
        self.assertTrue(parse_llm_range(raw, 100.0).get('fallback'))

    def test_last_valid_object_wins_in_plain_text(self):
        raw = ('Example: {"buy_low":1,"buy_high":2,"sell_low":3,"sell_high":4,"rationale":"e"}. '
               'Answer: {"buy_low":95,"buy_high":97,"sell_low":105,"sell_high":108,"rationale":"a"}')
        self.assertEqual(parse_llm_range(raw, 100.0)['rationale'], 'a')

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
        # Low breached buy_high * stop_threshold (default 0.95 = 95.0)
        # Use a low that is below 95 to trigger the stop
        delta = evaluate_range_prediction(high_price=101, low_price=93, pred=self.PRED)
        self.assertEqual(delta, -0.01)

    def test_stop_threshold_respected(self):
        # low_price=97 is NOT below 100*0.95=95 -> should be 0.0 (inconclusive)
        delta = evaluate_range_prediction(high_price=101, low_price=97, pred=self.PRED)
        self.assertEqual(delta, 0.0)

    def test_stop_threshold_custom(self):
        # With stop_threshold=0.98: low_price=97 IS below 100*0.98=98 -> stop
        delta = evaluate_range_prediction(high_price=101, low_price=97, pred=self.PRED, stop_threshold=0.98)
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
        # Fallback stop: would be -0.01 with explicit threshold, but should be 0.0
        fallback_pred = {**self.PRED, 'fallback': True}
        delta = evaluate_range_prediction(high_price=101, low_price=93, pred=fallback_pred, stop_threshold=0.95)
        self.assertEqual(delta, 0.0)


# ---------------------------------------------------------------------------
# score_deltas_from_journal
# ---------------------------------------------------------------------------

class ScoreDeltasFromJournalTests(unittest.TestCase):

    SCORES = {'model_a': 5.0, 'model_b': 5.0}

    def _trade(self, outcome, date, models, pnl_pct=1.0):
        return {'close_date': date, 'outcome': outcome, 'predicting_models': models, 'pnl_pct': pnl_pct}

    def test_win_adds_positive_delta(self):
        journal = [self._trade('win', '2026-05-20', ['model_a'])]
        deltas = score_deltas_from_journal(journal, '2026-05-20', self.SCORES)
        self.assertIn('model_a', deltas)
        self.assertEqual(deltas['model_a'], [0.02])

    def test_loss_adds_negative_delta(self):
        journal = [self._trade('loss', '2026-05-20', ['model_b'])]
        deltas = score_deltas_from_journal(journal, '2026-05-20', self.SCORES)
        self.assertEqual(deltas['model_b'], [-0.02])

    def test_realized_pnl_scales_delta(self):
        journal = [self._trade('win', '2026-05-20', ['model_a'], pnl_pct=2.5)]
        deltas = score_deltas_from_journal(journal, '2026-05-20', self.SCORES)
        self.assertEqual(deltas['model_a'], [0.05])

    def test_large_loss_is_capped(self):
        journal = [self._trade('loss', '2026-05-20', ['model_a'], pnl_pct=-8.0)]
        deltas = score_deltas_from_journal(journal, '2026-05-20', self.SCORES)
        self.assertEqual(deltas['model_a'], [-0.10])

    def test_custom_scale_and_cap(self):
        journal = [self._trade('win', '2026-05-20', ['model_a'], pnl_pct=5.0)]
        deltas = score_deltas_from_journal(
            journal, '2026-05-20', self.SCORES, scale=0.01, cap=0.03
        )
        self.assertEqual(deltas['model_a'], [0.03])

    def test_trade_closed_after_the_window_is_not_counted(self):
        journal = [self._trade('win', '2026-05-21', ['model_a'])]
        deltas = score_deltas_from_journal(journal, '2026-05-20', self.SCORES, after_date='2026-05-19')
        self.assertNotIn('model_a', deltas)

    def test_window_covers_days_without_a_predictions_file(self):
        # Previous scored file 05-21, this one 05-31: trades closed 05-22 and
        # 05-26 have no file of their own and used to be skipped for good.
        journal = [self._trade('win', '2026-05-22', ['model_a']),
                   self._trade('win', '2026-05-26', ['model_a']),
                   self._trade('win', '2026-05-31', ['model_a'])]
        deltas = score_deltas_from_journal(journal, '2026-05-31', self.SCORES, after_date='2026-05-21')
        self.assertEqual(deltas['model_a'], [0.02, 0.02, 0.02])

    def test_window_start_is_exclusive_so_each_trade_is_scored_once(self):
        # A trade closed on the previous window's end date was already scored.
        journal = [self._trade('win', '2026-05-21', ['model_a'])]
        deltas = score_deltas_from_journal(journal, '2026-05-31', self.SCORES, after_date='2026-05-21')
        self.assertNotIn('model_a', deltas)

    def test_adjacent_windows_tile_without_gaps_or_overlap(self):
        journal = [self._trade('win', f'2026-05-{d:02d}', ['model_a']) for d in range(18, 29)]
        seen = []
        for after, through in [('2026-05-17', '2026-05-20'), ('2026-05-20', '2026-05-26'), ('2026-05-26', '2026-05-28')]:
            seen += score_deltas_from_journal(journal, through, self.SCORES, after_date=after).get('model_a', [])
        self.assertEqual(len(seen), len(journal))

    def test_no_lower_bound_when_after_date_is_none(self):
        journal = [self._trade('win', '2026-05-01', ['model_a']), self._trade('win', '2026-05-20', ['model_a'])]
        deltas = score_deltas_from_journal(journal, '2026-05-20', self.SCORES)
        self.assertEqual(len(deltas['model_a']), 2)

    def test_unreliable_journal_rows_are_not_scored(self):
        # A qty-desync row records -48% for what was really a winning trade;
        # a capped -0.10 would punish the models for a bookkeeping bug.
        bad = {**self._trade('loss', '2026-05-20', ['model_a'], pnl_pct=-48.0), 'qty_desync_corrected': True}
        good = self._trade('win', '2026-05-20', ['model_a'])
        deltas = score_deltas_from_journal([bad, good], '2026-05-20', self.SCORES)
        self.assertEqual(deltas['model_a'], [0.02])

    def test_model_not_in_scores_ignored(self):
        journal = [self._trade('win', '2026-05-20', ['unknown_model'])]
        deltas = score_deltas_from_journal(journal, '2026-05-20', self.SCORES)
        self.assertNotIn('unknown_model', deltas)


class BarIsAfterPriorTests(unittest.TestCase):

    @staticmethod
    def _bar(day):
        # Alpaca stamps daily bars at midnight ET: 04:00 UTC in summer.
        return MagicMock(timestamp=datetime(2026, 10, day, 4, 0, tzinfo=timezone.utc))

    def test_a_later_session_is_gradable(self):
        self.assertTrue(bar_is_after_prior(self._bar(22), '2026-10-21'))

    def test_the_same_or_an_earlier_session_is_not(self):
        # Grading a prediction against the bar it was made from would score
        # the model on data it was shown.
        self.assertFalse(bar_is_after_prior(self._bar(21), '2026-10-21'))
        self.assertFalse(bar_is_after_prior(self._bar(20), '2026-10-21'))

    def test_no_prior_predictions_does_not_crash(self):
        # prior_date is None after a gap over 10 days or in a fresh sandbox:
        # `str <= None` raised a TypeError that would fail every nightly.
        self.assertTrue(bar_is_after_prior(self._bar(22), None))


class FetchRecentNewsTests(unittest.TestCase):
    """The prompt's news used to be empty for every ticker for weeks: Yahoo's
    news endpoint returned 404 and yfinance turned that into an empty list."""

    ET_STAMP = datetime(2026, 10, 30, 19, 0, tzinfo=timezone.utc)   # 15:00 ET on 10-30

    def setUp(self):
        import forge_loop
        self.fl = forge_loop
        fetch_recent_news.cache_clear()
        self.addCleanup(fetch_recent_news.cache_clear)
        patch.object(forge_loop, '_alpaca_news_client', return_value=object()).start()
        self.addCleanup(patch.stopall)

    def _alpaca(self, items=None, raises=None):
        if raises:
            return patch.object(self.fl.alpaca_client, 'fetch_news', side_effect=raises)
        return patch.object(self.fl.alpaca_client, 'fetch_news', return_value=items or [])

    def _yahoo(self, news=None, raises=None):
        search = MagicMock()
        if raises:
            search.side_effect = raises
        else:
            search.return_value.news = news or []
        return patch.object(self.fl.yf, 'Search', search), search

    def _item(self, n, source='Benzinga'):
        return {'headline': f'Headline {n}', 'source': source, 'created_at': self.ET_STAMP}

    def test_alpaca_headlines_are_formatted_with_source_and_et_date(self):
        yahoo, search = self._yahoo()
        with self._alpaca([self._item(1), self._item(2, 'Reuters')]), yahoo:
            out = fetch_recent_news('KO')
        self.assertEqual(out, '- Headline 1 (Benzinga, 10-30)\n- Headline 2 (Reuters, 10-30)')
        search.assert_not_called()

    def test_at_most_five_headlines(self):
        yahoo, _ = self._yahoo()
        with self._alpaca([self._item(i) for i in range(9)]), yahoo:
            self.assertEqual(len(fetch_recent_news('KO').splitlines()), 5)

    def test_falls_back_to_yahoo_search_when_alpaca_has_nothing(self):
        yahoo, search = self._yahoo(news=[
            {'title': 'Old format', 'publisher': 'Yahoo Finance'},
            {'content': {'title': 'New format', 'provider': {'displayName': 'Reuters'}}},
        ])
        with self._alpaca([]), yahoo:
            out = fetch_recent_news('KO')
        self.assertEqual(out, '- Old format (Yahoo Finance)\n- New format (Reuters)')
        search.assert_called_once_with('KO', news_count=5)

    def test_falls_back_when_alpaca_raises(self):
        yahoo, _ = self._yahoo(news=[{'title': 'From Yahoo', 'publisher': 'P'}])
        with self._alpaca(raises=RuntimeError('401 unauthorized')), yahoo:
            self.assertEqual(fetch_recent_news('KO'), '- From Yahoo (P)')

    def test_both_sources_failing_gives_the_default_text_never_an_exception(self):
        yahoo, _ = self._yahoo(raises=RuntimeError('boom'))
        with self._alpaca(raises=RuntimeError('boom')), yahoo:
            self.assertEqual(fetch_recent_news('KO'), 'No recent news available.')
        fetch_recent_news.cache_clear()
        yahoo, _ = self._yahoo(news=[])
        with self._alpaca([]), yahoo:
            self.assertEqual(fetch_recent_news('KO'), 'No recent news available.')

    def test_a_hung_alpaca_call_is_abandoned_and_the_fallback_is_used(self):
        import time
        real = self.fl._with_timeout
        patch.object(self.fl, '_with_timeout', lambda fn, timeout_s=0.2, default=None: real(fn, 0.2, default)).start()
        yahoo, _ = self._yahoo(news=[{'title': 'Fallback', 'publisher': 'P'}])
        with patch.object(self.fl.alpaca_client, 'fetch_news', side_effect=lambda *a, **k: time.sleep(30)), yahoo:
            start = time.time()
            out = fetch_recent_news('KO')
        self.assertEqual(out, '- Fallback (P)')
        self.assertLess(time.time() - start, 5)

    def test_cached_per_ticker_so_both_models_share_one_fetch(self):
        yahoo, _ = self._yahoo()
        with patch.object(self.fl.alpaca_client, 'fetch_news', return_value=[self._item(1)]) as fetch, yahoo:
            first = fetch_recent_news('KO')
            second = fetch_recent_news('KO')
            fetch_recent_news('NVDA')
        self.assertEqual(first, second)
        self.assertEqual(fetch.call_count, 2)    # KO once, NVDA once


class PreviousPredictionDateTests(unittest.TestCase):

    def _prev(self, files, before):
        import forge_loop
        with tempfile.TemporaryDirectory() as d:
            for day in files:
                open(os.path.join(d, f'predictions_{day}.json'), 'w').close()
            with patch.object(forge_loop, 'HISTORY_DIR', d):
                return previous_prediction_date(before)

    def test_newest_file_strictly_before_the_date(self):
        self.assertEqual(self._prev(['2026-05-21', '2026-05-31', '2026-06-02'], '2026-05-31'), '2026-05-21')

    def test_ignores_files_on_or_after_the_date(self):
        self.assertEqual(self._prev(['2026-06-02', '2026-06-03'], '2026-06-02'), '2026-06-01')

    def test_no_earlier_file_falls_back_to_the_day_before_not_everything(self):
        # An unbounded window would rescore every old trade on every run.
        self.assertEqual(self._prev([], '2026-05-31'), '2026-05-30')


class TradeScoreDeltaTests(unittest.TestCase):

    def test_trade_score_delta_uses_realized_pnl(self):
        trade = {'outcome': 'win', 'pnl_pct': 1.75}
        self.assertEqual(_trade_score_delta(trade), 0.035)

    def test_trade_score_delta_normalizes_inconsistent_loss_sign(self):
        trade = {'outcome': 'loss', 'pnl_pct': 2.0}
        self.assertEqual(_trade_score_delta(trade), -0.04)

    def test_trade_score_delta_is_bounded(self):
        trade = {'outcome': 'win', 'pnl_pct': 20.0}
        self.assertEqual(_trade_score_delta(trade), 0.10)


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

    def test_score_decay_shrinks_the_evidence_before_adjustment(self):
        # decay=0.9 from 8.0: 5 + (8 - 5) * 0.9 + 0.01 = 7.71; no decay: 8.01
        decayed, nodecay = {'model_a': 8.0}, {'model_a': 8.0}
        apply_score_deltas(decayed, {'model_a': [0.01]}, decay=0.9)
        apply_score_deltas(nodecay, {'model_a': [0.01]}, decay=1.0)
        self.assertAlmostEqual(decayed['model_a'], 7.71, places=3)
        self.assertAlmostEqual(nodecay['model_a'], 8.01, places=3)

    def test_decay_pulls_toward_neutral_from_both_sides_not_toward_zero(self):
        scores = {'above': 8.0, 'below': 2.0, 'neutral': 5.0}
        apply_score_deltas(scores, {}, decay=0.9)
        self.assertAlmostEqual(scores['above'], 7.7, places=3)
        self.assertAlmostEqual(scores['below'], 2.3, places=3)
        self.assertEqual(scores['neutral'], 5.0)

    def test_model_with_nothing_scored_today_still_decays(self):
        # It used to be skipped, so an idle model kept its score forever.
        scores = {'idle': 8.0, 'active': 8.0}
        apply_score_deltas(scores, {'active': [0.01], 'idle': [0.0, 0.0]}, decay=0.9)
        self.assertAlmostEqual(scores['idle'], 7.7, places=3)
        self.assertAlmostEqual(scores['active'], 7.71, places=3)

    def test_steady_evidence_settles_near_neutral_plus_adjustment_over_one_minus_decay(self):
        # +0.01 a day with decay 0.99 settles at 5 + 0.01 / 0.01 = 6.0. The old
        # formula settled at 0.01 / 0.01 = 1.0, below where it started.
        scores = {'model_a': 5.0}
        for _ in range(2000):
            apply_score_deltas(scores, {'model_a': [0.01]}, decay=0.99)
        self.assertAlmostEqual(scores['model_a'], 6.0, places=1)
        scores = {'model_a': 5.0}
        for _ in range(2000):
            apply_score_deltas(scores, {'model_a': [-0.01]}, decay=0.99)
        self.assertAlmostEqual(scores['model_a'], 4.0, places=1)

    def test_a_real_loss_is_not_diluted_by_many_small_checks(self):
        # 50 price checks at +0.01 and one stopped trade at -0.10. Averaged
        # together (the old behavior) this was a net gain; separately the
        # real loss counts in full.
        scores = {'model_a': 5.0}
        apply_score_deltas(scores, {'model_a': [0.01] * 50}, trade_deltas_by_model={'model_a': [-0.10]})
        self.assertAlmostEqual(scores['model_a'], 5.0 + 0.01 - 0.10, places=3)

    def test_trade_deltas_are_averaged_not_summed(self):
        scores = {'model_a': 5.0}
        apply_score_deltas(scores, {}, trade_deltas_by_model={'model_a': [0.02, 0.04, -0.03]})
        self.assertAlmostEqual(scores['model_a'], 5.0 + (0.02 + 0.04 - 0.03) / 3, places=3)

    def test_model_only_present_in_deltas_is_created_at_neutral(self):
        scores = {'model_a': 5.0}
        apply_score_deltas(scores, {'model_b': [0.01]}, decay=0.5)
        self.assertAlmostEqual(scores['model_b'], 5.01, places=3)
        self.assertEqual(list(scores), ['model_a', 'model_b'])

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


class FormatRecentOhlcTests(unittest.TestCase):

    def test_empty_bars_returns_empty_string(self):
        self.assertEqual(format_recent_ohlc([]), "")

    def test_includes_only_last_n_rows(self):
        bars = _make_bars(list(range(100, 120)))  # 20 bars, closes 100..119
        out = format_recent_ohlc(bars, n=5)
        # The 5 most recent closes are 115..119; 114 must be excluded.
        self.assertIn('119.00', out)
        self.assertIn('115.00', out)
        self.assertNotIn('114.00', out)

    def test_renders_date_and_volume(self):
        from datetime import datetime
        bars = [
            MockBar(100.0, 101.0, 99.0, volume=2_500_000, open=99.5,
                    timestamp=datetime(2026, 5, 20)),
            MockBar(102.0, 103.0, 100.5, volume=900_000, open=100.0,
                    timestamp=datetime(2026, 5, 21)),
        ]
        out = format_recent_ohlc(bars)
        self.assertIn('2026-05-20', out)
        self.assertIn('2026-05-21', out)
        self.assertIn('2.5M', out)   # millions formatting
        self.assertIn('900K', out)   # thousands formatting
        self.assertIn('Open', out)   # header present

    def test_oldest_first_ordering(self):
        from datetime import datetime
        bars = [
            MockBar(100.0, timestamp=datetime(2026, 5, 20)),
            MockBar(105.0, timestamp=datetime(2026, 5, 21)),
        ]
        out = format_recent_ohlc(bars)
        self.assertLess(out.index('2026-05-20'), out.index('2026-05-21'))


# ---------------------------------------------------------------------------
# _with_timeout
#
# Regression coverage for 2026-08-20: a yfinance call (.news / .calendar)
# hung indefinitely on a network stall, and forge_loop's own bare
# try/except Exception around it does nothing for a hang -- the whole
# nightly run sat silent for 2h21m producing zero data, only ending when
# the runner service was externally restarted.
# ---------------------------------------------------------------------------

class WithTimeoutTests(unittest.TestCase):

    def test_returns_default_when_call_hangs_past_timeout(self):
        import time

        def hangs():
            time.sleep(30)
            return 'never'

        start = time.time()
        result = _with_timeout(hangs, timeout_s=0.2, default='fallback')
        elapsed = time.time() - start

        self.assertEqual(result, 'fallback')
        # Must actually return promptly, not block for anywhere near the
        # hung call's real duration.
        self.assertLess(elapsed, 5)

    def test_passes_through_normal_return_value(self):
        result = _with_timeout(lambda: 42, timeout_s=5, default=None)
        self.assertEqual(result, 42)

    def test_returns_default_when_call_raises(self):
        def boom():
            raise RuntimeError('network error')

        result = _with_timeout(boom, timeout_s=5, default='fallback')
        self.assertEqual(result, 'fallback')

    def test_abandoned_thread_does_not_block_process_exit(self):
        # The bug in an earlier draft of this fix: using
        # concurrent.futures.ThreadPoolExecutor's worker threads (which are
        # NOT daemon threads) meant the interpreter's own atexit machinery
        # still joined the hung thread before the *process* could exit, even
        # though _with_timeout itself returned promptly. Verify the timed-out
        # call really is a daemon thread, so the process is never blocked at
        # shutdown by an abandoned network call.
        import threading
        import time

        def hangs():
            time.sleep(30)

        threads_before = set(threading.enumerate())
        _with_timeout(hangs, timeout_s=0.1, default=None)
        new_threads = set(threading.enumerate()) - threads_before
        self.assertTrue(new_threads, "expected the worker thread to still be running")
        for t in new_threads:
            self.assertTrue(t.daemon, f"{t} must be a daemon thread")


class FetchAllBarsTests(unittest.TestCase):

    def test_end_is_a_timestamp_so_the_session_just_closed_is_included(self):
        # Alpaca reads a date-only end as 00:00 UTC of that day. The nightly
        # runs at 23:00 UTC, the same UTC date as the session that just
        # closed, so a date-only end silently dropped it and every run saw
        # the previous day's bar (identical closes in the 08-23 and 08-24
        # prediction files).
        import forge_loop
        data_client = MagicMock()
        data_client.get_stock_bars.return_value.data = {}
        with patch.object(forge_loop.alpaca_client, 'get_data_client', return_value=data_client):
            fetch_all_bars(['NVDA'])

        req = data_client.get_stock_bars.call_args.args[0]
        # alpaca-py normalizes end to naive UTC; a date-only end would be
        # today's midnight, hours before the nightly run.
        end_utc = req.end.replace(tzinfo=req.end.tzinfo or timezone.utc)
        self.assertLess(abs(datetime.now(timezone.utc) - end_utc), timedelta(minutes=1))


class FindLatestPredictionsPathTests(unittest.TestCase):

    def test_newest_file_strictly_before_the_run_date(self):
        import forge_loop
        with tempfile.TemporaryDirectory() as d:
            for day in ('2026-09-24', '2026-09-25', '2026-09-28'):
                open(os.path.join(d, f'predictions_{day}.json'), 'w').close()
            with patch.object(forge_loop, 'HISTORY_DIR', d):
                # Monday's run grades Friday's file, never its own.
                path, date_str = forge_loop.find_latest_predictions_path('2026-09-28')
        self.assertEqual(date_str, '2026-09-25')
        self.assertTrue(path.endswith('predictions_2026-09-25.json'))

    def test_none_when_nothing_in_lookback(self):
        import forge_loop
        with tempfile.TemporaryDirectory() as d:
            with patch.object(forge_loop, 'HISTORY_DIR', d):
                self.assertEqual(forge_loop.find_latest_predictions_path('2026-09-28'), (None, None))


class BarSessionDateTests(unittest.TestCase):

    def test_midnight_et_stamps_map_to_their_session_in_both_dst_states(self):
        import forge_loop
        edt = MagicMock(timestamp=datetime(2026, 9, 28, 4, 0, tzinfo=timezone.utc))
        est = MagicMock(timestamp=datetime(2026, 11, 30, 5, 0, tzinfo=timezone.utc))
        self.assertEqual(forge_loop.bar_session_date(edt), '2026-09-28')
        self.assertEqual(forge_loop.bar_session_date(est), '2026-11-30')


if __name__ == '__main__':
    unittest.main()
