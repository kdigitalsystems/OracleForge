"""Tests for scripts/generate_html_report.py (pure helpers, builders, full run)."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import tempfile
import unittest

ROOT = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    'generate_html_report', os.path.join(ROOT, 'scripts', 'generate_html_report.py'))
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


def _trade(**kw):
    t = {'ticker': 'AAA', 'entry_price': 10.0, 'exit_price': 11.0, 'pnl_pct': 10.0,
         'pnl_usd': 1.0, 'usd_invested': 10.0, 'outcome': 'win', 'close_date': '2026-10-01'}
    t.update(kw)
    return t


@contextlib.contextmanager
def tempcwd():
    old = os.getcwd()
    with tempfile.TemporaryDirectory() as d:
        os.chdir(d)
        try:
            yield d
        finally:
            os.chdir(old)


def _write(path, obj):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        f.write(obj if isinstance(obj, str) else json.dumps(obj))


class HelperTests(unittest.TestCase):
    def test_js_json_escapes_script_close(self):
        out = gen._js_json(['</script><script>alert(1)</script>', 'a&b'])
        self.assertNotIn('<', out)
        self.assertNotIn('>', out)
        self.assertNotIn('&', out)
        self.assertEqual(json.loads(out), ['</script><script>alert(1)</script>', 'a&b'])

    def test_safe_numbers(self):
        self.assertEqual(gen._num(None), 0.0)
        self.assertEqual(gen._num('abc', 5), 5)
        self.assertEqual(gen._num('2.5'), 2.5)
        self.assertEqual(gen._num(float('nan')), 0.0)
        self.assertIsNone(gen._num_or_none(None))
        self.assertIsNone(gen._num_or_none('x'))
        self.assertEqual(gen._num_or_none('0.5'), 0.5)
        self.assertEqual(gen._str(None), '')
        self.assertEqual(gen._str(3), '3')

    def test_money_negative(self):
        self.assertEqual(gen._money(-196.84), '-$196.84')
        self.assertEqual(gen._money(-196.84, 4, True), '-$196.8400')
        self.assertEqual(gen._money(3, 2, True), '+$3.00')
        self.assertEqual(gen._fmt_usd(-1.5), '-$1.5000')
        self.assertEqual(gen._fmt_usd(1.5), '$1.5000')
        self.assertEqual(gen._fmt_usd(None), '-')
        self.assertNotIn('$-', gen._fmt_usd(-0.01))

    def test_score_bar_clamp(self):
        self.assertEqual(gen._bar_pct(15), 100.0)
        self.assertEqual(gen._bar_pct(-3), 0.0)
        self.assertEqual(gen._bar_pct(None), 0.0)
        self.assertAlmostEqual(gen._bar_pct(5), 50.0)
        html = gen.build_scores_section({'a': 15, 'b': None, 'c': -2})
        self.assertNotIn('width:150', html)
        self.assertNotIn('width:-', html)

    def test_load_json_tolerant(self):
        with tempcwd():
            _write('empty.json', '')
            _write('bad.json', '{oops')
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.assertEqual(gen.load_json('missing.json', 'd'), 'd')
                self.assertEqual(gen.load_json('empty.json', 'd'), 'd')
                self.assertEqual(gen.load_json('bad.json', 'd'), 'd')
            self.assertIn('WARNING', err.getvalue())


class SignalsTests(unittest.TestCase):
    def test_zero_upside_renders(self):
        html = gen.build_signals_section({'active': [{'ticker': 'ZZ', 'upside_pct': 0.0}]}, '2026-10-13')
        self.assertIn('+0.00%', html)

    def test_weird_values_do_not_raise(self):
        rows = [{'ticker': 'A', 'consensus_cv': 'high', 'upside_pct': 'n/a'},
                {'ticker': None, 'consensus_cv': None, 'upside_pct': None}]
        html = gen.build_signals_section({'active': rows, 'generated_at': None}, '2026-10-13')
        self.assertIn('n/a', html)

    def test_generated_text(self):
        self.assertIn('runner clock', gen._generated_text('2026-10-14T08:22:20'))
        self.assertIn('UTC', gen._generated_text('2026-10-14T08:22:20Z'))
        self.assertNotIn('T08', gen._generated_text('2026-10-14T08:22:20'))
        self.assertIn('Generated', gen._generated_text('garbage'))


class BannerTests(unittest.TestCase):
    def _banner(self, preds):
        with tempcwd():
            old = gen.HISTORY_DIR
            gen.HISTORY_DIR = 'history'
            try:
                if preds is not None:
                    _write('history/predictions_2026-10-13.json', preds)
                return gen.build_forecast_banner('2026-10-13')
            finally:
                gen.HISTORY_DIR = old

    @staticmethod
    def _preds(flags):
        # Real shape: a fallback prediction has fallback=True, a real one has
        # no 'fallback' key at all.
        return {f'T{i}': {'models': {'m': ({'buy_high': 1.0, 'fallback': True} if f else {'buy_high': 1.0})}}
                for i, f in enumerate(flags)}

    def test_real_predictions_without_a_fallback_key_count_toward_the_total(self):
        # 261 real + 107 fallback + 10 earnings-skipped, as in the 2026-10-13 file.
        preds = {f'R{i}': {'models': {'m': {'buy_high': 1.0}}} for i in range(261)}
        preds.update({f'F{i}': {'models': {'m': {'fallback': True}}} for i in range(107)})
        preds.update({f'S{i}': {'models': {'m': {'skipped': True, 'reason': 'upcoming_earnings'}}} for i in range(10)})
        self.assertEqual(self._banner(preds), '')

    def test_threshold(self):
        b = self._banner(self._preds([True, False]))
        self.assertIn('Forecast degraded: 1 of 2', b)
        self.assertIn('amber', b)
        self.assertEqual(self._banner(self._preds([True, False, False])), '')
        self.assertIn('3 of 3', self._banner(self._preds([True] * 3)))

    def test_skipped_and_odd_shapes(self):
        preds = self._preds([True])
        preds['S'] = {'models': {'m': {'skipped': True, 'reason': 'x'}}}
        preds['N'] = {'models': None}
        preds['X'] = 'junk'
        self.assertIn('1 of 1', self._banner(preds))

    def test_missing_or_empty(self):
        self.assertEqual(self._banner(None), '')
        self.assertEqual(self._banner({}), '')
        self.assertEqual(self._banner([1, 2]), '')
        self.assertEqual(self._banner({'A': {'models': {}}}), '')


class PnlTests(unittest.TestCase):
    def test_reliable_only_and_note(self):
        good = [_trade(pnl_usd=2.0, pnl_pct=5.0, close_date='2026-10-01'),
                _trade(pnl_usd=-1.0, pnl_pct=-10.0, exit_price=9.0, outcome='loss', close_date='2026-10-02', ticker='BBB')]
        bad = [_trade(qty_desync_corrected=True, pnl_usd=-100.0, pnl_pct=-50.0, ticker='BAD1', outcome='loss'),
               _trade(qty_desync_corrected=True, pnl_usd=-108.14, pnl_pct=-50.0, ticker='BAD2', outcome='loss')]
        html = gen.build_pnl_section(good + bad)
        self.assertIn('+$1.0000', html)          # total from reliable only
        self.assertIn('2 earlier trades with unreliable accounting', html)
        self.assertIn('-$208.14', html)
        self.assertNotIn('BAD1', html)
        self.assertNotIn('BAD2', html)
        self.assertIn('Best trade', html)
        self.assertIn('Worst trade', html)
        self.assertIn('50%', html)

    def test_no_note_when_clean(self):
        html = gen.build_pnl_section([_trade()])
        self.assertNotIn('unreliable', html)

    def test_empty_and_all_flagged(self):
        self.assertIn('No closed trades yet', gen.build_pnl_section([]))
        html = gen.build_pnl_section([_trade(qty_desync_corrected=True, pnl_usd=-5.0)])
        self.assertIn('No trades with reliable accounting', html)
        self.assertIn('1 earlier trade with unreliable accounting', html)
        self.assertIn('-$5.00', html)

    def test_nulls(self):
        rows = [_trade(pnl_usd=None, pnl_pct=None, close_date=None),
                _trade(close_date='2026-10-02')]
        html = gen.build_pnl_section(rows)
        self.assertIn('Recent trades', html)

    def test_negative_total_format(self):
        html = gen.build_pnl_section([_trade(pnl_usd=-196.84, pnl_pct=-10.0, exit_price=9.0, outcome='loss')])
        self.assertIn('-$196.8400', html)
        self.assertNotIn('$-', html)


class OtherSectionTests(unittest.TestCase):
    def test_backtest_nulls(self):
        report = {'strategy': {'win_rate': None, 'profit_factor': 'x', 'trades': None,
                               'by_exit_reason': {'stop': {'trades': 1, 'total_pnl_usd': None}}},
                  'by_model': {'m1': {'win_rate': None, 'avg_return_pct': None, 'profit_factor': 'bad'},
                               'm2': {'avg_return_pct': 1.0}},
                  'params': {'order_usd': None, 'stop_loss_pct': 'x'}}
        self.assertIn('Backtest', gen.build_backtest_section(report))

    def test_walk_forward_without_train(self):
        report = {'summary': {}, 'latest_recommendation': {'selected_params': {'a': 1}}}
        html = gen.build_walk_forward_section(report)
        self.assertIn('Latest pick', html)

    def test_attribution_excluded(self):
        base = {'trades': 3, 'overall': {'win_rate': None}, 'by_model': {'m': {'win_rate': None}}}
        self.assertNotIn('excluded', gen.build_attribution_section(base))
        base['excluded'] = {'count': 0}
        self.assertNotIn('excluded', gen.build_attribution_section(base))
        base['excluded'] = {'count': 7, 'recorded_pnl_usd': -1.0}
        self.assertIn('7 trades with unreliable accounting excluded', gen.build_attribution_section(base))

    def test_table_scope(self):
        self.assertIn('<th scope="col"', gen._table(['A'], [['x']]))


class GenerateTests(unittest.TestCase):
    def test_nasty_inputs(self):
        evil = '</script><script>alert(1)</script>'
        with tempcwd():
            _write('reports/signals_2026-10-13.json', {
                'generated_at': '2026-10-14T08:22:20', 'summary': {'total': None},
                'active': [{'ticker': evil, 'upside_pct': 0.0, 'consensus_cv': 'x'},
                           {'ticker': 'OK', 'upside_pct': 'bad', 'consensus_cv': None}],
                'skip': [{'ticker': None, 'signal': None}], 'stale': []})
            _write('history/predictions_2026-10-13.json',
                   {'A': {'models': {'m': {'fallback': True}, 'n': {'fallback': True}}}})
            _write('history/trade_journal.json', [
                _trade(ticker=evil, pnl_usd=None, pnl_pct=None, close_date=None),
                _trade(close_date='2026-10-03', pnl_usd=-2.0, pnl_pct=-10.0, exit_price=9.0, outcome='loss'),
                _trade(qty_desync_corrected=True, pnl_usd=-9.0),
                'junk', None])
            _write('state/analyst_scores.json', {'a': None, evil: 99, 'b': 'x'})
            _write('reports/backtest_summary.json', {'strategy': {'win_rate': None}, 'by_model': {'m': None}})
            _write('reports/walk_forward_summary.json', '{not json')
            _write('reports/trade_attribution.json',
                   {'trades': None, 'by_model': {'m': {'win_rate': None}}, 'excluded': {'count': 'many'}})
            err = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
                gen.generate()
            with open('docs/index.html', encoding='utf-8') as f:
                html = f.read()
        self.assertNotIn('</script><script>alert', html)
        self.assertNotIn('<script>alert', html)
        self.assertIn('Forecast degraded: 2 of 2', html)
        self.assertIn('WARNING', err.getvalue())
        self.assertIn('role="tablist"', html)
        self.assertIn('Escape', html)

    def test_empty_run(self):
        with tempcwd():
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                gen.generate()
            self.assertTrue(os.path.exists('docs/index.html'))


class PageSecurityTests(unittest.TestCase):

    def _page(self):
        with tempcwd():
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                gen.generate()
            with open('docs/index.html', encoding='utf-8') as f:
                return f.read()

    def test_rebuild_button_asks_for_a_fine_grained_actions_only_token(self):
        html = self._page()
        self.assertIn('settings/personal-access-tokens/new', html)
        self.assertIn('Actions: Read and write', html)
        self.assertNotIn('settings/tokens/new', html)
        self.assertNotIn('scopes=workflow', html)
        self.assertNotIn('ghp_', html)

    def test_third_party_script_with_a_stable_url_has_an_integrity_hash(self):
        html = self._page()
        tag = [t for t in html.split('<script') if 'chart.umd.min.js' in t][0]
        self.assertIn('integrity="sha384-', tag)
        self.assertIn('crossorigin="anonymous"', tag)


if __name__ == '__main__':
    unittest.main()
