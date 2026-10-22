"""Unit tests for scripts/model_canary.py (the pre-flight model check)."""
import contextlib
import importlib.util
import io
import json
import os
import subprocess
import tempfile
import unittest
from types import SimpleNamespace

_spec = importlib.util.spec_from_file_location(
    'model_canary', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts', 'model_canary.py'))
canary = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(canary)

REAL = {'buy_low': 1.0, 'buy_high': 1.1, 'sell_low': 1.2, 'sell_high': 1.3}   # no 'fallback' key
FALLBACK = {**REAL, 'fallback': True}


def _entry(**models):
    return {'models': models}


class FallbackReportTests(unittest.TestCase):

    def test_real_predictions_have_no_fallback_key_and_still_count(self):
        stats, bad = canary.fallback_report({'A': _entry(m=REAL), 'B': _entry(m=REAL), 'C': _entry(m=FALLBACK)})
        self.assertEqual((stats['m']['total'], stats['m']['fallback']), (3, 1))
        self.assertEqual(bad, [])

    def test_exactly_half_is_allowed_more_than_half_is_not(self):
        half = {'A': _entry(m=REAL), 'B': _entry(m=FALLBACK)}
        self.assertEqual(canary.fallback_report(half)[1], [])
        most = {'A': _entry(m=REAL), 'B': _entry(m=FALLBACK), 'C': _entry(m=FALLBACK)}
        self.assertEqual(canary.fallback_report(most)[1], ['m'])

    def test_each_model_is_judged_on_its_own(self):
        preds = {t: _entry(good=REAL, bad=FALLBACK) for t in 'ABCD'}
        self.assertEqual(canary.fallback_report(preds)[1], ['bad'])

    def test_earnings_skipped_entries_are_not_counted(self):
        preds = {'A': _entry(m={'skipped': True, 'reason': 'upcoming_earnings'}), 'B': _entry(m=REAL)}
        self.assertEqual(canary.fallback_report(preds)[0]['m']['total'], 1)

    def test_odd_shapes_do_not_raise(self):
        self.assertEqual(canary.fallback_report(None), ({}, []))
        self.assertEqual(canary.fallback_report({'A': None, 'B': {'models': None}, 'C': {'models': {'m': 'x'}}}), ({}, []))


def _fake_repo(root, models=('good', 'bad')):
    for rel, body in [('config/models.json', list(models)), ('config/tickers.json', ['KO']),
                      ('state/analyst_scores.json', {m: 5.0 for m in models}), ('history/trade_journal.json', [])]:
        path = os.path.join(root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(body, f)
    os.makedirs(os.path.join(root, 'reports'))
    with open(os.path.join(root, 'reports', 'must_not_be_copied.json'), 'w') as f:
        f.write('{}')
    with open(os.path.join(root, 'history', 'predictions_2026-10-16.json'), 'w') as f:
        f.write('{"REAL": "committed forecast"}')


def _runner(predictions=None, returncode=0, raises=None, log='', record=None):
    def run(cmd, cwd, **kwargs):
        if record is not None:
            record.update(cwd=cwd, cmd=cmd, kwargs=kwargs)
        if raises:
            raise raises
        if predictions is not None:
            os.makedirs(os.path.join(cwd, 'history'), exist_ok=True)
            with open(os.path.join(cwd, 'history', 'predictions_2026-10-22.json'), 'w') as f:
                json.dump(predictions, f)
        return SimpleNamespace(returncode=returncode, stdout=log, stderr='')
    return run


class RunCanaryTests(unittest.TestCase):

    def _run(self, **runner_kwargs):
        with tempfile.TemporaryDirectory() as repo:
            _fake_repo(repo)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = canary.run_canary('KO,NVDA', 5, runner=_runner(**runner_kwargs), repo=repo)
            return code, out.getvalue(), repo

    def _healthy(self):
        return {t: _entry(good=REAL, bad=REAL) for t in ('KO', 'NVDA')}

    def test_healthy_when_every_model_forecasts(self):
        code, out, _ = self._run(predictions=self._healthy())
        self.assertEqual(code, 0)
        self.assertIn('HEALTHY', out)

    def test_unhealthy_when_a_model_falls_back_on_most_tickers(self):
        preds = {t: _entry(good=REAL, bad=FALLBACK) for t in ('KO', 'NVDA')}
        code, out, _ = self._run(predictions=preds, log="  [!] bad output for KO unparseable; using fallback. Raw: 'garbage'")
        self.assertEqual(code, 1)
        self.assertIn("UNHEALTHY: ['bad']", out)
        self.assertIn('garbage', out)

    def test_unhealthy_when_a_configured_model_made_no_prediction(self):
        code, out, _ = self._run(predictions={t: _entry(good=REAL) for t in ('KO', 'NVDA')})
        self.assertEqual(code, 1)
        self.assertIn("no prediction at all from ['bad']", out)

    def test_unhealthy_when_forge_exits_nonzero_or_writes_nothing_or_times_out(self):
        self.assertEqual(self._run(predictions=self._healthy(), returncode=2, log='boom')[0], 1)
        self.assertEqual(self._run(predictions=None)[0], 1)
        self.assertEqual(self._run(raises=subprocess.TimeoutExpired('forge', 1))[0], 1)

    def test_runs_in_a_sandbox_and_leaves_the_repo_untouched(self):
        record = {}
        with tempfile.TemporaryDirectory() as repo:
            _fake_repo(repo)
            before = sorted(os.listdir(os.path.join(repo, 'history')))
            with contextlib.redirect_stdout(io.StringIO()):
                canary.run_canary('KO', 5, runner=_runner(predictions=self._healthy(), record=record), repo=repo)
            self.assertNotEqual(os.path.realpath(record['cwd']), os.path.realpath(repo))
            self.assertEqual(sorted(os.listdir(os.path.join(repo, 'history'))), before)
            self.assertEqual(open(os.path.join(repo, 'history', 'predictions_2026-10-16.json')).read(),
                             '{"REAL": "committed forecast"}')
        self.assertFalse(os.path.exists(record['cwd']), 'sandbox should be removed afterwards')
        self.assertEqual(record['kwargs']['timeout'], 300)
        self.assertIn('--tickers', record['cmd'])

    def test_sandbox_gets_only_what_forge_loop_reads(self):
        with tempfile.TemporaryDirectory() as repo, tempfile.TemporaryDirectory() as sandbox:
            _fake_repo(repo)
            canary.prepare_sandbox(sandbox, repo)
            found = sorted(os.path.relpath(os.path.join(d, f), sandbox) for d, _, fs in os.walk(sandbox) for f in fs)
            self.assertEqual(found, sorted([os.path.join('config', 'models.json'), os.path.join('config', 'tickers.json'),
                                            os.path.join('state', 'analyst_scores.json'),
                                            os.path.join('history', 'trade_journal.json')]))


if __name__ == '__main__':
    unittest.main()
