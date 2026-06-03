# backtest.py
"""Walk prediction history and score range outcomes vs next-session OHLC."""
from __future__ import annotations

import argparse
import math
import os
import statistics
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
BACKTEST_HISTORY_DIR = 'reports/backtest_history/'
TRADING_CONFIG_FILE = 'config/trading.json'
DEFAULT_STOP_LOSS_PCT = 0.95
MIN_ADEQUATE_SAMPLE = 30   # trades below this -> treat metrics as not yet trustworthy
T_CRIT_95 = 1.96           # ~95% two-sided (normal approx; conservative for small n)


def load_stop_loss_pct() -> float:
    """Read stop_loss_pct from the trading config so the backtest matches live trading."""
    cfg = load_json(TRADING_CONFIG_FILE, {})
    return float(cfg.get('stop_loss_pct', DEFAULT_STOP_LOSS_PCT))


def _stats(returns: list[float]) -> dict:
    """Summarise a list of per-trade returns with a significance read.

    Answers the core measurement question: is the mean return distinguishable
    from zero, and do we even have enough trades to say? t_stat uses a normal
    approximation; for small n it is only indicative, which is exactly why
    `sample_adequate` is reported alongside.
    """
    n = len(returns)
    if n == 0:
        return {'n': 0, 'mean': 0.0, 'std': 0.0, 't_stat': None,
                'ci95_low': None, 'ci95_high': None,
                'significant': False, 'sample_adequate': False}
    mean = sum(returns) / n
    std = statistics.stdev(returns) if n >= 2 else 0.0
    stderr = std / math.sqrt(n) if (std > 0 and n >= 2) else 0.0
    t_stat = (mean / stderr) if stderr > 0 else None
    half = T_CRIT_95 * stderr
    return {
        'n': n,
        'mean': round(mean, 4),
        'std': round(std, 4),
        't_stat': round(t_stat, 3) if t_stat is not None else None,
        'ci95_low': round(mean - half, 4),
        'ci95_high': round(mean + half, 4),
        # "significant" = mean return's 95% CI excludes 0 AND we have enough trades
        'significant': bool(t_stat is not None and abs(t_stat) > T_CRIT_95 and n >= MIN_ADEQUATE_SAMPLE),
        'sample_adequate': n >= MIN_ADEQUATE_SAMPLE,
    }


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

    # Normalise to tz-naive dates so the comparison works regardless of
    # whether yfinance returns a tz-aware or tz-naive DatetimeIndex.
    after_ts = pd.Timestamp(after_date).normalize()
    index_norm = hist.index.normalize()
    if index_norm.tz is not None:
        index_norm = index_norm.tz_localize(None)
    future = hist[index_norm > after_ts]
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


def simulate_range_outcome(bar: dict, pred: dict, stop_loss_pct: float = 0.95) -> dict:
    """
    Simulate a buy/sell range trade against a realized OHLC bar.

    Entry assumed at buy_high (conservative). The stop is placed at
    buy_high * stop_loss_pct to match the live trader and the nightly scorer,
    both of which use the same configured stop_loss_pct (default 0.95 = -5%).
    Path within the bar is unknown, so the stop is checked before the target
    (pessimistic assumption).
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
    stop = entry * stop_loss_pct

    if low <= stop:
        return {'outcome': 'stop', 'return_pct': round((stop_loss_pct - 1) * 100, 2), 'triggered': True}

    if high >= sell_low:
        ret = ((sell_low - entry) / entry) * 100
        return {'outcome': 'win', 'return_pct': round(ret, 2), 'triggered': True}

    ret = ((high - entry) / entry) * 100
    return {'outcome': 'miss', 'return_pct': round(ret, 2), 'triggered': True}


def _new_bucket() -> dict:
    return {
        'trades': 0,
        'triggered': 0,
        'wins': 0,       # categorical: reached the sell target
        'stops': 0,      # categorical: hit the stop
        'misses': 0,     # categorical: triggered but neither target nor stop
        'return_pct_sum': 0.0,
        'gross_profit': 0.0,   # sum of all positive returns
        'gross_loss': 0.0,     # sum of abs of all negative returns
        '_win_n': 0,           # count of trades with positive return
        '_loss_n': 0,          # count of trades with negative return
        '_cur_loss_streak': 0,
        '_max_loss_streak': 0,
        '_returns': [],        # per-triggered-trade returns, for significance
    }


def _update_bucket(bucket: dict, outcome: dict) -> None:
    bucket['trades'] += 1
    if not outcome['triggered']:
        return

    bucket['triggered'] += 1
    ret = outcome['return_pct']
    bucket['return_pct_sum'] += ret
    bucket['_returns'].append(ret)

    # Categorical outcome counts drive win_rate.
    if outcome['outcome'] == 'win':
        bucket['wins'] += 1
    elif outcome['outcome'] == 'stop':
        bucket['stops'] += 1
    else:
        bucket['misses'] += 1

    # P&L aggregation is by the sign of the realized return, so the numerator
    # and denominator of avg_win_pct / avg_loss_pct always cover the same
    # population (a "miss" that closed below entry counts as a loss).
    if ret > 0:
        bucket['gross_profit'] += ret
        bucket['_win_n'] += 1
        bucket['_cur_loss_streak'] = 0
    elif ret < 0:
        bucket['gross_loss'] += abs(ret)
        bucket['_loss_n'] += 1
        bucket['_cur_loss_streak'] += 1
        bucket['_max_loss_streak'] = max(
            bucket['_max_loss_streak'], bucket['_cur_loss_streak']
        )
    # ret == 0 (scratch): no P&L, streak unchanged.


def _finalize_bucket(bucket: dict) -> dict:
    triggered = bucket['triggered']
    wins = bucket['wins']
    win_n = bucket['_win_n']
    loss_n = bucket['_loss_n']
    gross_profit = bucket['gross_profit']
    gross_loss = bucket['gross_loss']

    avg_win_pct = round(gross_profit / win_n, 4) if win_n else 0.0
    avg_loss_pct = round(gross_loss / loss_n, 4) if loss_n else 0.0
    profit_factor = round(gross_profit / gross_loss, 3) if gross_loss > 0 else None

    # Build output, excluding internal tracking fields
    out = {k: v for k, v in bucket.items() if not k.startswith('_')}
    out.update({
        'win_rate': round(wins / triggered, 4) if triggered else 0.0,
        'avg_return_pct': round(bucket['return_pct_sum'] / triggered, 4) if triggered else 0.0,
        'avg_win_pct': avg_win_pct,
        'avg_loss_pct': avg_loss_pct,
        'profit_factor': profit_factor,
        'max_consecutive_losses': bucket['_max_loss_streak'],
        # Is the mean trade return distinguishable from zero, given the sample?
        'significance': _stats(bucket['_returns']),
    })
    return out


def run_backtest(
    dates: list[str] | None = None,
    scores: dict | None = None,
    stop_loss_pct: float | None = None,
) -> dict:
    scores = scores or load_json(SCORES_FILE, {})
    config = load_signal_config()
    dates = dates or list_prediction_dates()
    if stop_loss_pct is None:
        stop_loss_pct = load_stop_loss_pct()

    by_model: dict[str, dict] = defaultdict(_new_bucket)
    by_signal: dict[str, dict] = defaultdict(_new_bucket)
    daily_rows = []
    skipped = 0

    # Benchmark: does following ACTIVE signals beat just buying the same names
    # and holding to the next close? Paired per (ticker, date).
    active_strategy_rets: list[float] = []
    active_buyhold_rets: list[float] = []
    universe_buyhold_rets: list[float] = []

    for pred_date in dates:
        path = os.path.join(HISTORY_DIR, f'predictions_{pred_date}.json')
        predictions = load_json(path, {})
        day_detail = {'date': pred_date, 'tickers_evaluated': 0}
        day_consensus_rets: list[float] = []

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

            # Buy-and-hold baseline for this name: prior close -> next close.
            prior_close = closes.get(ticker)
            buyhold_ret = None
            if prior_close and prior_close > 0:
                buyhold_ret = (bar['close'] - prior_close) / prior_close * 100
                universe_buyhold_rets.append(buyhold_ret)

            consensus = weighted_consensus_ranges(model_preds, scores)

            for model_name, pred in model_preds.items():
                outcome = simulate_range_outcome(bar, pred, stop_loss_pct)
                _update_bucket(by_model[model_name], outcome)

            if consensus:
                consensus_outcome = simulate_range_outcome(bar, consensus, stop_loss_pct)
                _update_bucket(by_signal['CONSENSUS'], consensus_outcome)
                _update_bucket(by_signal[signal], consensus_outcome)
                day_consensus_rets.append(consensus_outcome['return_pct'])

                # ACTIVE signal vs buy-and-hold, paired on the same name/date.
                # Strategy holds cash (0%) when the limit entry never triggers.
                if signal == 'ACTIVE' and buyhold_ret is not None:
                    strat_ret = consensus_outcome['return_pct'] if consensus_outcome['triggered'] else 0.0
                    active_strategy_rets.append(strat_ret)
                    active_buyhold_rets.append(buyhold_ret)

        if day_detail['tickers_evaluated'] > 0:
            day_detail['consensus_avg_return_pct'] = (
                round(sum(day_consensus_rets) / len(day_consensus_rets), 4)
                if day_consensus_rets else 0.0
            )
            daily_rows.append(day_detail)

    # Running cumulative of the per-day average consensus return (equity proxy).
    cum = 0.0
    for row in daily_rows:
        cum += row.get('consensus_avg_return_pct', 0.0)
        row['cumulative_return_pct'] = round(cum, 4)

    active_edge = [s - b for s, b in zip(active_strategy_rets, active_buyhold_rets)]

    return {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'prediction_dates': dates,
        'days_in_history': len(dates),
        'stop_loss_pct': stop_loss_pct,
        'skipped_pairs': skipped,
        'by_model': {name: _finalize_bucket(b) for name, b in sorted(by_model.items())},
        'by_signal': {name: _finalize_bucket(b) for name, b in sorted(by_signal.items())},
        'benchmark': {
            'active_strategy': _stats(active_strategy_rets),
            'active_buy_hold': _stats(active_buyhold_rets),
            'active_edge_vs_buy_hold': _stats(active_edge),
            'universe_buy_hold': _stats(universe_buyhold_rets),
        },
        'daily': daily_rows,
    }


def print_backtest_summary(report: dict) -> None:
    print('\n=== OracleForge Backtest Summary ===')
    print(
        f"Prediction files: {report['days_in_history']} | "
        f"Skipped (no data): {report['skipped_pairs']}"
    )

    print('\nBy model:')
    print(
        f"{'Model':<36} {'Trades':>8} {'Win%':>7} "
        f"{'AvgWin%':>9} {'AvgLoss%':>9} {'PF':>7} {'MaxLoss':>8}"
    )
    print('-' * 90)
    for model, stats in report['by_model'].items():
        pf = f"{stats['profit_factor']:.2f}" if stats['profit_factor'] is not None else 'N/A'
        print(
            f"{model:<36} {stats['trades']:>8} "
            f"{stats['win_rate'] * 100:>6.1f}% "
            f"{stats['avg_win_pct']:>8.2f}% "
            f"{stats['avg_loss_pct']:>8.2f}% "
            f"{pf:>7} "
            f"{stats['max_consecutive_losses']:>8}"
        )

    print('\nBy signal:')
    print(
        f"{'Signal':<12} {'Trades':>8} {'Win%':>7} "
        f"{'AvgWin%':>9} {'AvgLoss%':>9} {'PF':>7} {'AvgRet%':>9} {'Signif?':>9}"
    )
    print('-' * 78)
    for signal, stats in report['by_signal'].items():
        pf = f"{stats['profit_factor']:.2f}" if stats['profit_factor'] is not None else 'N/A'
        sig = stats.get('significance', {})
        flag = 'YES' if sig.get('significant') else ('thin' if not sig.get('sample_adequate') else 'no')
        print(
            f"{signal:<12} {stats['trades']:>8} "
            f"{stats['win_rate'] * 100:>6.1f}% "
            f"{stats['avg_win_pct']:>8.2f}% "
            f"{stats['avg_loss_pct']:>8.2f}% "
            f"{pf:>7} "
            f"{stats['avg_return_pct']:>8.3f}% "
            f"{flag:>9}"
        )

    # --- Significance & benchmark: the "can we trust this / does it beat
    #     doing nothing" read that the aggregate ratios alone can't give. ---
    bench = report.get('benchmark', {})
    if bench:
        def _line(label, s):
            if not s or s.get('n', 0) == 0:
                print(f"  {label:<26} (no data)")
                return
            ci = f"[{s['ci95_low']:+.3f}, {s['ci95_high']:+.3f}]"
            adq = 'ok' if s['sample_adequate'] else f'THIN (<{MIN_ADEQUATE_SAMPLE})'
            print(f"  {label:<26} n={s['n']:<5} mean={s['mean']:+.3f}%  95%CI {ci}  sample={adq}")

        print('\nBenchmark — ACTIVE signal vs. buy-and-hold (same names, prior close -> next close):')
        _line('ACTIVE strategy', bench.get('active_strategy', {}))
        _line('ACTIVE buy & hold', bench.get('active_buy_hold', {}))
        _line('Edge (strategy - hold)', bench.get('active_edge_vs_buy_hold', {}))
        _line('Universe buy & hold', bench.get('universe_buy_hold', {}))

        edge = bench.get('active_edge_vs_buy_hold', {})
        print('\nVerdict:')
        if not edge or not edge.get('sample_adequate'):
            print(f"  NOT ENOUGH DATA yet (need >= {MIN_ADEQUATE_SAMPLE} ACTIVE trades for a read). "
                  "Treat all metrics above as indicative only.")
        elif edge.get('significant') and edge.get('mean', 0) > 0:
            print(f"  ACTIVE signals beat buy-and-hold by {edge['mean']:+.3f}%/trade "
                  "(statistically significant).")
        elif edge.get('significant') and edge.get('mean', 0) < 0:
            print(f"  ACTIVE signals UNDERPERFORM buy-and-hold by {edge['mean']:+.3f}%/trade "
                  "(significant) — the signal layer is hurting returns.")
        else:
            print(f"  No significant edge over buy-and-hold (mean {edge.get('mean', 0):+.3f}%/trade, "
                  "not distinguishable from zero). The signal isn't adding measurable value yet.")


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

    # Archive a timestamped copy so metric evolution is trackable over time.
    os.makedirs(BACKTEST_HISTORY_DIR, exist_ok=True)
    stamp = datetime.now().strftime('%Y-%m-%d')
    archive_path = os.path.join(BACKTEST_HISTORY_DIR, f'backtest_{stamp}.json')
    save_json(archive_path, report)

    print_backtest_summary(report)
    print(f"\nFull report saved to {out_path}")
    print(f"Archived snapshot to {archive_path}")


if __name__ == '__main__':
    main()
