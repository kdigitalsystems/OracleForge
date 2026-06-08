#!/usr/bin/env python3
"""Validate OracleForge CI outputs before git commit."""
import json
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

CONFIG_TICKERS = 'config/tickers.json'
HISTORY_DIR = 'history/'
REPORTS_DIR = 'reports/'
ET = ZoneInfo('America/New_York')


def fail(message: str) -> None:
    print(f"ERROR: {message}")
    sys.exit(1)


def main() -> None:
    if not os.path.exists(CONFIG_TICKERS):
        fail(f"Missing {CONFIG_TICKERS}")

    with open(CONFIG_TICKERS, 'r') as f:
        tickers = json.load(f)

    if not isinstance(tickers, list) or len(tickers) == 0:
        fail(f"{CONFIG_TICKERS} is empty — forge_loop would not produce data")

    today = datetime.now(ET).strftime('%Y-%m-%d')
    predictions_path = os.path.join(HISTORY_DIR, f'predictions_{today}.json')
    signals_path = os.path.join(REPORTS_DIR, f'signals_{today}.json')

    if not os.path.exists(predictions_path):
        fail(f"Missing today's predictions file: {predictions_path}")

    with open(predictions_path, 'r') as f:
        predictions = json.load(f)

    if not predictions:
        fail(f"{predictions_path} is empty")

    sample = next(iter(predictions.values()), {})
    required = {'models', 'close', 'signal'}
    if not isinstance(sample, dict) or not required.issubset(sample.keys()):
        fail("Predictions file is not enriched (missing required fields: models, close, signal)")

    if not os.path.exists(signals_path):
        fail(f"Missing today's signals report: {signals_path}")

    with open(signals_path, 'r') as f:
        report = json.load(f)

    summary = report.get('summary', {})
    print(
        f"OK: {len(tickers)} tickers configured, "
        f"{len(predictions)} predictions, "
        f"{summary.get('active', 0)} ACTIVE setups"
    )


if __name__ == '__main__':
    main()
