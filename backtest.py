# backtest.py
"""Walk prediction history and score range outcomes vs next-session OHLC."""
from __future__ import annotations

import argparse
import os
from collections import defaultdict
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from signals import (
    HISTORY_DIR,
    REPORTS_DIR,
    SCORES_FILE,
    build_enriched_predictions,
    extract_model_predictions,
    list_prediction_dates,
    load_json,
    load_signal_config,
    save_json,
    weighted_consensus_ranges,
)

BACKTEST_DIR = 'reports/'


def fetch_next_session_bar(ticker: str, after_date: str, max_days: int = 10) -> dict | None:
    """Return OHLC for the first trading session strictly after after_date."""
    start = datetime.strptime(after_date, '%Y-%m-%d')
    end = start + timedelta(days=max_days)
    try:
        hist = yf.Ticker(ticker).history(
            start=after_date,
            end=end.strftime('%Y-%m-%d'),
            auto_adjust=False,
        )
    except Exception:
        return None

    if hist.empty:
        return None

    after_ts = pd.Timestamp(after_date)
    future = hist[hist.index.normalize() > after_ts.normalize()]
    if future.empty:
        return None

    row = future.iloc[0]
    return {
        'date': future.index[0].strftime('%Y-%m-%d'),
        'open': float(row['Open']),
        'high': float(row['High']),
        'low': float(row['Low']),
        'close': float(row['Close']),
    }


def simulate_range_outcome(bar: dict, pred: dict) -> dict:
    """
    Simulate a buy/sell range trade against a realized OHLC bar.
    Entry assumed at buy_high (conservative); stop at -2% from entry.
    """
    buy_high = float(pred.get('buy_high') or 0)
    sell_low = float(pred.get('sell_low') or 0)

    if buy_high <= 0 or sell_low <= buy_high:
        return {'outcome': 'invalid', 'return_pct': 0.0, 'triggered': False}

    low = bar['low']
    high = bar['high']

    if low > buy_high:
        return {'outcome': 'no_trigger', 'return_pct': 0.0, 'triggered': False}

    entry = buy_high
    stop = entry * 0.98

    if low <= stop:
        return {'outcome': 'stop', 'return_pct': -2.0, 'triggered': True}

    if high >= sell_low:
        ret = ((sell_low - entry) / entry) * 100
        return {'outcome': 'win', 'return_pct': round(ret, 2), 'triggered': True}

    ret = ((high - entry) / entry) * 100
    return {'outcome': 'miss', 'return_pct': round(ret, 2), 'triggered': True}


def _new_bucket() -> dict:
    return {
        'trades': 0,
        'triggered': 0,
        'wins': 0,
        'stops': 0,
        'misses': 0,
        'return_pct_sum': 0.0,
    }


def _update_bucket(bucket: dict, outcome: dict) -> None:
    bucket['trades'] += 1
    if outcome['triggered']:
        bucket['triggered'] += 1
        bucket['return_pct_sum'] += outcome['return_pct']
        if outcome['outcome'] == 'win':
            bucket['wins'] += 1
        elif outcome['outcome'] == 'stop':
            bucket['stops'] += 1
        else:
            bucket['misses'] += 1


def _finalize_bucket(bucket: dict) -> dict:
    triggered = bucket['triggered']
    return {
        **bucket,
        'win_rate': round(bucket['wins'] / triggered, 4) if triggered else 0.0,
        'avg_return_pct': round(bucket['return_pct_sum'] / triggered, 4) if triggered else 0.0,
    }


def run_backtest(dates: list[str] | None = None, scores: dict | None = None) -> dict:
    scores = scores or load_json(SCORES_FILE, {})
    config = load_signal_config()
    dates = dates or list_prediction_dates()

    by_model: dict[str, dict] = defaultdict(_new_bucket)
    by_signal: dict[str, dict] = defaultdict(_new_bucket)
    daily_rows = []
    skipped = 0

    for pred_date in dates:
        path = os.path.join(HISTORY_DIR, f'predictions_{pred_date}.json')
        predictions = load_json(path, {})
        day_detail = {'date': pred_date, 'tickers_evaluated': 0}

        closes = {}
        raw = {}
        for ticker, entry in predictions.items():
            if isinstance(entry, dict) and 'close' in entry:
                closes[ticker] = float(entry['close'])
            model_preds = extract_model_predictions(entry)
            if model_preds:
                raw[ticker] = model_preds

        enriched = {}
        if closes and raw:
            enriched = build_enriched_predictions(raw, closes, scores, config)

        for ticker, entry in predictions.items():
            model_preds = extract_model_predictions(entry)
            if not model_preds:
                continue

            bar = fetch_next_session_bar(ticker, pred_date)
            if bar is None:
                skipped += 1
                continue

            day_detail['tickers_evaluated'] += 1
            signal = enriched.get(ticker, {}).get('signal', 'SKIP')

            consensus = weighted_consensus_ranges(model_preds, scores)

            for model_name, pred in model_preds.items():
                outcome = simulate_range_outcome(bar, pred)
                _update_bucket(by_model[model_name], outcome)

            if consensus:
                _update_bucket(by_signal['CONSENSUS'], simulate_range_outcome(bar, consensus))
                _update_bucket(by_signal[signal], simulate_range_outcome(bar, consensus))

        if day_detail['tickers_evaluated'] > 0:
            daily_rows.append(day_detail)

    return {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'prediction_dates': dates,
        'days_in_history': len(dates),
        'skipped_pairs': skipped,
        'by_model': {name: _finalize_bucket(b) for name, b in sorted(by_model.items())},
        'by_signal': {name: _finalize_bucket(b) for name, b in sorted(by_signal.items())},
        'daily': daily_rows,
    }


def print_backtest_summary(report: dict) -> None:
    print('\n=== OracleForge Backtest Summary ===')
    print(
        f"Prediction files: {report['days_in_history']} | "
        f"Skipped (no data): {report['skipped_pairs']}"
    )

    print('\nBy model:')
    print(f"{'Model':<36} {'Trades':>8} {'Triggered':>10} {'Win%':>8} {'AvgRet%':>10}")
    print('-' * 76)
    for model, stats in report['by_model'].items():
        print(
            f"{model:<36} {stats['trades']:>8} {stats['triggered']:>10} "
            f"{stats['win_rate'] * 100:>7.1f}% {stats['avg_return_pct']:>9.2f}%"
        )

    print('\nBy signal:')
    print(f"{'Signal':<12} {'Trades':>8} {'Triggered':>10} {'Win%':>8} {'AvgRet%':>10}")
    print('-' * 52)
    for signal, stats in report['by_signal'].items():
        print(
            f"{signal:<12} {stats['trades']:>8} {stats['triggered']:>10} "
            f"{stats['win_rate'] * 100:>7.1f}% {stats['avg_return_pct']:>9.2f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description='Backtest OracleForge predictions against realized OHLC.')
    parser.add_argument('--from-date', help='Start date YYYY-MM-DD (inclusive)')
    parser.add_argument('--to-date', help='End date YYYY-MM-DD (inclusive)')
    args = parser.parse_args()

    dates = list_prediction_dates()
    if args.from_date:
        dates = [d for d in dates if d >= args.from_date]
    if args.to_date:
        dates = [d for d in dates if d <= args.to_date]

    if not dates:
        print('No prediction history files found in history/.')
        return

    print(f"Backtesting {len(dates)} day(s): {dates[0]} .. {dates[-1]}")
    report = run_backtest(dates)

    os.makedirs(BACKTEST_DIR, exist_ok=True)
    out_path = os.path.join(BACKTEST_DIR, 'backtest_summary.json')
    save_json(out_path, report)
    print_backtest_summary(report)
    print(f"\nFull report saved to {out_path}")


if __name__ == '__main__':
    main()
