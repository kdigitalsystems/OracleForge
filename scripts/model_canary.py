#!/usr/bin/env python3
"""Pre-flight: do both models give usable forecasts on THIS machine right now?

Runs the real forecast (production prompt, every configured model) on a few
tickers inside a throwaway sandbox, then fails if any model fell back on more
than half of them. Nothing in the repo working tree is touched: forge_loop
runs with the sandbox as its working directory, which holds copies of the
config, scores and journal it needs and receives everything it writes.

Exists because deepseek-r1:8b returned gibberish on server1 for weeks while
every other check passed, and the nightly happily committed ~95% fallback
forecasts. The nightly runs this before the real forecast so a bad runner
fails fast (and the retry watchdog can try again) instead of publishing bad
signals; runner_health.yml uses it for the canary.

Exit codes: 0 healthy, 1 unhealthy or the forecast could not run.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TICKERS = 'KO,NVDA,F,XLE'
MAX_FALLBACK_RATE = 0.5   # a model falling back on more than half is unhealthy
DEFAULT_TIMEOUT_MIN = 25
# What forge_loop reads from the working directory; copied into the sandbox.
SANDBOX_FILES = ['config', 'state', os.path.join('history', 'trade_journal.json')]


def fallback_report(predictions: dict) -> tuple[dict, list[str]]:
    """Per-model fallback counts and the models over the allowed rate.

    Real predictions carry no 'fallback' key; fallback ones have it set to
    True. Earnings-skipped entries have no prediction and are not counted.
    """
    total: collections.Counter = collections.Counter()
    fell_back: collections.Counter = collections.Counter()
    for entry in (predictions or {}).values():
        for model, pred in ((entry or {}).get('models') or {}).items():
            if isinstance(pred, dict) and not pred.get('skipped'):
                total[model] += 1
                fell_back[model] += bool(pred.get('fallback'))
    stats = {m: {'total': total[m], 'fallback': fell_back[m], 'rate': fell_back[m] / total[m]} for m in sorted(total)}
    bad = [m for m, s in stats.items() if s['rate'] > MAX_FALLBACK_RATE]
    return stats, bad


def prepare_sandbox(dest: str, repo: str = REPO) -> None:
    for rel in SANDBOX_FILES:
        src = os.path.join(repo, rel)
        target = os.path.join(dest, rel)
        if os.path.isdir(src):
            shutil.copytree(src, target)
        elif os.path.exists(src):
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copy2(src, target)


def run_canary(tickers: str = DEFAULT_TICKERS, timeout_min: int = DEFAULT_TIMEOUT_MIN,
               runner=subprocess.run, repo: str = REPO) -> int:
    expected = [t for t in tickers.split(',') if t]
    with tempfile.TemporaryDirectory(prefix='oracleforge-canary-') as sandbox:
        prepare_sandbox(sandbox, repo)
        cmd = [sys.executable, os.path.join(repo, 'forge_loop.py'), '--tickers', ','.join(expected),
               '--batch-size', str(len(expected))]
        print(f'canary: forecasting {expected} in a throwaway sandbox ({timeout_min} min limit)', flush=True)
        try:
            proc = runner(cmd, cwd=sandbox, capture_output=True, text=True, timeout=timeout_min * 60)
        except subprocess.TimeoutExpired:
            print(f'UNHEALTHY: the forecast did not finish within {timeout_min} minutes.')
            return 1
        log = (proc.stdout or '') + (proc.stderr or '')
        if proc.returncode != 0:
            print(f'UNHEALTHY: forge_loop exited {proc.returncode}. Last output:')
            print('\n'.join(log.splitlines()[-15:]))
            return 1
        files = sorted(glob.glob(os.path.join(sandbox, 'history', 'predictions_*.json')))
        if not files:
            print('UNHEALTHY: forge_loop produced no predictions file.')
            print('\n'.join(log.splitlines()[-15:]))
            return 1
        with open(files[-1]) as f:
            predictions = json.load(f)
    stats, bad = fallback_report(predictions)
    print(f'canary tickers forecast: {len(predictions)} of {len(expected)}')
    for model, s in stats.items():
        print(f"  {model}: {s['fallback']}/{s['total']} fell back ({100 * s['rate']:.0f}%)")
    for line in [l for l in log.splitlines() if 'unparseable' in l][:6]:
        print('  ' + line.strip()[:260])
    configured = _configured_models(repo)
    silent = [m for m in configured if m not in stats]
    if silent:
        print(f'UNHEALTHY: no prediction at all from {silent}.')
        return 1
    if bad:
        print(f'UNHEALTHY: {bad} fell back on more than {MAX_FALLBACK_RATE:.0%} of the tickers on this runner.')
        return 1
    print('HEALTHY: every model produced usable ranges on this runner.')
    return 0


def _configured_models(repo: str) -> list[str]:
    try:
        with open(os.path.join(repo, 'config', 'models.json')) as f:
            return list(json.load(f))
    except (OSError, ValueError):
        return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--tickers', default=DEFAULT_TICKERS, help='Comma-separated tickers to forecast.')
    parser.add_argument('--timeout-min', type=int, default=DEFAULT_TIMEOUT_MIN)
    args = parser.parse_args()
    return run_canary(args.tickers, args.timeout_min)


if __name__ == '__main__':
    sys.exit(main())
