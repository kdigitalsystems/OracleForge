# backtest.py
"""Walk prediction history and score range outcomes vs next-session OHLC."""
from __future__ import annotations

import argparse
import itertools
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
WALK_FORWARD_FILE = 'reports/walk_forward_summary.json'
ATTRIBUTION_FILE = 'reports/trade_attribution.json'
TRADING_CONFIG_FILE = 'config/trading.json'
DEFAULT_STOP_LOSS_PCT = 0.95
MIN_ADEQUATE_SAMPLE = 30   # trades below this -> treat metrics as not yet trustworthy
T_CRIT_95 = 1.96           # ~95% two-sided (normal approx; conservative for small n)
DEFAULT_WALK_FORWARD_GRID = {
    'min_upside_pct': [1.0, 1.5, 2.0],
    'max_consensus_cv': [0.05, 0.10, 0.15],
    'stop_loss_pct': [0.95, 0.97, 0.98],
    'execution': ['limit_stop', 'limit_hold'],
}


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


_BAR_CACHE: dict[tuple[str, str], dict | None] = {}


def fetch_next_session_bar(ticker: str, after_date: str, max_days: int = 10) -> dict | None:
    """Return OHLC for the first trading session strictly after after_date.

    Memoized so running the backtest under several execution modes only hits
    yfinance once per (ticker, date).
    """
    key = (ticker, after_date)
    if key in _BAR_CACHE:
        return _BAR_CACHE[key]
    result = _fetch_next_session_bar_uncached(ticker, after_date, max_days)
    _BAR_CACHE[key] = result
    return result


def _fetch_next_session_bar_uncached(ticker: str, after_date: str, max_days: int = 10) -> dict | None:
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


EXECUTION_MODES = ('limit_stop', 'limit_hold', 'market_hold')


def simulate_range_outcome(bar: dict, pred: dict, stop_loss_pct: float = 0.95,
                           execution: str = 'limit_stop') -> dict:
    """
    Simulate a trade against a realized OHLC bar under one of three execution
    models, so we can A/B which mechanics actually help:

      limit_stop  (current live behavior): enter only if price dips to buy_high;
                  exit at sell_low (win), at buy_high*stop_loss_pct (stop), or
                  at the day's high (miss).
      limit_hold: enter only if price dips to buy_high; no stop/target — exit at
                  the close. Isolates whether the stop/target is the leak.
      market_hold: ignore the range entirely — buy at the open, sell at the
                  close. Isolates whether the *selection* (not the mechanics)
                  has any edge.
    """
    buy_high = float(pred.get('buy_high') or 0)
    sell_low = float(pred.get('sell_low') or 0)

    if buy_high <= 0 or sell_low <= buy_high:
        return {'outcome': 'invalid', 'return_pct': 0.0, 'triggered': False}

    low = bar['low']
    high = bar['high']
    close = bar['close']
    open_ = bar.get('open') or close

    if execution == 'market_hold':
        ret = ((close - open_) / open_) * 100 if open_ else 0.0
        return {'outcome': 'win' if ret >= 0 else 'stop',
                'return_pct': round(ret, 2), 'triggered': True}

    # limit-entry modes only trigger if the day traded down to buy_high
    if low > buy_high:
        return {'outcome': 'no_trigger', 'return_pct': 0.0, 'triggered': False}

    entry = buy_high

    if execution == 'limit_hold':
        ret = ((close - entry) / entry) * 100
        return {'outcome': 'win' if ret >= 0 else 'stop',
                'return_pct': round(ret, 2), 'triggered': True}

    # default: limit_stop
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
    execution: str = 'limit_stop',
    config: dict | None = None,
) -> dict:
    scores = scores or load_json(SCORES_FILE, {})
    config = config or load_signal_config()
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
    # Selection diagnostic: buy-and-hold return grouped by the signal the
    # classifier assigned. If ACTIVE's buy&hold < SKIP's, selection is
    # anti-predictive independent of any execution mechanics.
    signal_buyhold: dict[str, list] = defaultdict(list)

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
                signal_buyhold[signal].append(buyhold_ret)

            consensus = weighted_consensus_ranges(model_preds, scores)

            for model_name, pred in model_preds.items():
                outcome = simulate_range_outcome(bar, pred, stop_loss_pct, execution)
                _update_bucket(by_model[model_name], outcome)

            if consensus:
                consensus_outcome = simulate_range_outcome(bar, consensus, stop_loss_pct, execution)
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
        'execution': execution,
        'skipped_pairs': skipped,
        'by_model': {name: _finalize_bucket(b) for name, b in sorted(by_model.items())},
        'by_signal': {name: _finalize_bucket(b) for name, b in sorted(by_signal.items())},
        'benchmark': {
            'active_strategy': _stats(active_strategy_rets),
            'active_buy_hold': _stats(active_buyhold_rets),
            'active_edge_vs_buy_hold': _stats(active_edge),
            'universe_buy_hold': _stats(universe_buyhold_rets),
        },
        # Selection quality: buy&hold return by assigned signal (mechanics-free).
        'selection': {sig: _stats(rets) for sig, rets in sorted(signal_buyhold.items())},
        'daily': daily_rows,
    }


def compare_executions(dates: list[str], scores: dict | None = None) -> dict:
    """Run the backtest under each execution model (bars are cached, so this is
    cheap) and return the ACTIVE strategy return + edge vs buy-and-hold for each.
    """
    results = {}
    for mode in EXECUTION_MODES:
        rep = run_backtest(dates, scores=scores, execution=mode)
        b = rep['benchmark']
        results[mode] = {
            'active_strategy': b['active_strategy'],
            'active_edge_vs_buy_hold': b['active_edge_vs_buy_hold'],
        }
    return results


def _grid_candidates(grid: dict | None = None) -> list[dict]:
    """Expand a small parameter grid into concrete candidate configs."""
    grid = grid or DEFAULT_WALK_FORWARD_GRID
    keys = list(grid.keys())
    return [
        dict(zip(keys, values))
        for values in itertools.product(*(grid[k] for k in keys))
    ]


def _candidate_report(dates: list[str], candidate: dict, scores: dict | None = None) -> dict:
    config = {**load_signal_config()}
    for key in ('min_upside_pct', 'max_consensus_cv', 'max_spread_pct', 'min_agreeing_models'):
        if key in candidate:
            config[key] = candidate[key]
    return run_backtest(
        dates,
        scores=scores,
        stop_loss_pct=float(candidate.get('stop_loss_pct', load_stop_loss_pct())),
        execution=candidate.get('execution', 'limit_stop'),
        config=config,
    )


def _score_candidate(report: dict) -> tuple[float, int, float]:
    """Rank candidates by validation edge, then sample size, then strategy return."""
    bench = report.get('benchmark', {})
    edge = bench.get('active_edge_vs_buy_hold', {})
    strat = bench.get('active_strategy', {})
    return (
        float(edge.get('mean') or 0.0),
        int(edge.get('n') or 0),
        float(strat.get('mean') or 0.0),
    )


def walk_forward_optimize(
    dates: list[str],
    train_window: int = 4,
    test_window: int = 1,
    grid: dict | None = None,
    scores: dict | None = None,
    runner=None,
) -> dict:
    """Pick params on prior windows, then validate on later unseen windows.

    This is intentionally non-mutating: it produces evidence and a latest
    recommendation, but never rewrites config/signals/trading settings.
    """
    candidates = _grid_candidates(grid)
    runner = runner or _candidate_report
    windows = []
    validation_edges: list[float] = []
    validation_returns: list[float] = []

    end = train_window
    while end < len(dates):
        train_dates = dates[end - train_window:end]
        test_dates = dates[end:end + test_window]
        if not test_dates:
            break

        ranked = []
        for candidate in candidates:
            train_report = runner(train_dates, candidate, scores)
            ranked.append((_score_candidate(train_report), candidate, train_report))
        ranked.sort(key=lambda row: row[0], reverse=True)

        _, best_candidate, train_report = ranked[0]
        test_report = runner(test_dates, best_candidate, scores)
        test_edge = test_report['benchmark']['active_edge_vs_buy_hold']
        test_strategy = test_report['benchmark']['active_strategy']
        edge_mean = float(test_edge.get('mean') or 0.0)
        strat_mean = float(test_strategy.get('mean') or 0.0)
        validation_edges.append(edge_mean)
        validation_returns.append(strat_mean)

        windows.append({
            'train_dates': train_dates,
            'test_dates': test_dates,
            'selected_params': best_candidate,
            'train_edge': train_report['benchmark']['active_edge_vs_buy_hold'],
            'test_edge': test_edge,
            'test_strategy': test_strategy,
        })
        end += test_window

    latest_recommendation = None
    if candidates and len(dates) >= train_window:
        latest_train = dates[-train_window:]
        latest_ranked = []
        for candidate in candidates:
            report = runner(latest_train, candidate, scores)
            latest_ranked.append((_score_candidate(report), candidate, report))
        latest_ranked.sort(key=lambda row: row[0], reverse=True)
        latest_recommendation = {
            'train_dates': latest_train,
            'selected_params': latest_ranked[0][1],
            'train_edge': latest_ranked[0][2]['benchmark']['active_edge_vs_buy_hold'],
        }

    return {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'train_window': train_window,
        'test_window': test_window,
        'candidate_count': len(candidates),
        'windows': windows,
        'summary': {
            'validation_windows': len(windows),
            'avg_validation_edge_pct': (
                round(sum(validation_edges) / len(validation_edges), 4)
                if validation_edges else 0.0
            ),
            'avg_validation_strategy_pct': (
                round(sum(validation_returns) / len(validation_returns), 4)
                if validation_returns else 0.0
            ),
            'positive_edge_windows': sum(1 for v in validation_edges if v > 0),
        },
        'latest_recommendation': latest_recommendation,
    }


def _avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def summarize_trade_attribution(journal: list[dict]) -> dict:
    """Break actual closed trades into model/ticker/execution-quality buckets."""
    model: dict[str, dict] = defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': []})
    ticker: dict[str, dict] = defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': []})
    execution = {
        'price_improvement_pct': [],
        'target_capture_pct': [],
        'stop_loss_count': 0,
        'profit_target_count': 0,
    }

    for trade in journal:
        try:
            pnl = float(trade.get('pnl_pct', 0.0))
            entry = float(trade.get('entry_price', 0.0))
            exit_ = float(trade.get('exit_price', 0.0))
            buy_high = float(trade.get('consensus_buy_high', 0.0))
            sell_low = float(trade.get('consensus_sell_low', 0.0))
        except (TypeError, ValueError):
            continue

        win = trade.get('outcome') == 'win'
        sym = trade.get('ticker', 'UNKNOWN')
        ticker[sym]['trades'] += 1
        ticker[sym]['wins'] += int(win)
        ticker[sym]['pnl'].append(pnl)

        for name in trade.get('predicting_models', []):
            model[name]['trades'] += 1
            model[name]['wins'] += int(win)
            model[name]['pnl'].append(pnl)

        if buy_high > 0 and entry > 0:
            execution['price_improvement_pct'].append(round((buy_high - entry) / buy_high * 100, 4))
        if sell_low > buy_high and exit_ > 0:
            execution['target_capture_pct'].append(round((exit_ - buy_high) / (sell_low - buy_high) * 100, 4))
        if sell_low > 0 and exit_ >= sell_low:
            execution['profit_target_count'] += 1
        elif not win:
            execution['stop_loss_count'] += 1

    def finalize(bucket: dict) -> dict:
        rows = {}
        for key, stats in bucket.items():
            n = stats['trades']
            rows[key] = {
                'trades': n,
                'wins': stats['wins'],
                'win_rate': round(stats['wins'] / n, 4) if n else 0.0,
                'avg_pnl_pct': _avg(stats['pnl']),
                'total_pnl_pct': round(sum(stats['pnl']), 4),
            }
        return dict(sorted(rows.items(), key=lambda item: item[1]['total_pnl_pct'], reverse=True))

    pnl_values = [float(t.get('pnl_pct', 0.0)) for t in journal if t.get('pnl_pct') is not None]
    wins = sum(1 for t in journal if t.get('outcome') == 'win')
    return {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'trades': len(journal),
        'overall': {
            'wins': wins,
            'win_rate': round(wins / len(journal), 4) if journal else 0.0,
            'avg_pnl_pct': _avg(pnl_values),
            'total_pnl_pct': round(sum(pnl_values), 4),
        },
        'by_model': finalize(model),
        'by_ticker': finalize(ticker),
        'execution_quality': {
            'avg_price_improvement_pct': _avg(execution['price_improvement_pct']),
            'avg_target_capture_pct': _avg(execution['target_capture_pct']),
            'profit_target_count': execution['profit_target_count'],
            'stop_loss_count': execution['stop_loss_count'],
        },
    }


def print_walk_forward_summary(report: dict) -> None:
    summary = report['summary']
    print('\n=== Walk-Forward Optimizer ===')
    print(
        f"Windows: {summary['validation_windows']} | "
        f"Avg validation edge: {summary['avg_validation_edge_pct']:+.3f}% | "
        f"Positive windows: {summary['positive_edge_windows']}"
    )
    rec = report.get('latest_recommendation')
    if rec:
        print(f"Latest recommended params from {rec['train_dates'][0]}..{rec['train_dates'][-1]}:")
        for key, value in rec['selected_params'].items():
            print(f"  {key}: {value}")


def print_attribution_summary(report: dict) -> None:
    print('\n=== Trade Attribution ===')
    overall = report['overall']
    print(
        f"Trades: {report['trades']} | Win rate: {overall['win_rate'] * 100:.1f}% | "
        f"Avg P&L: {overall['avg_pnl_pct']:+.3f}% | Total P&L: {overall['total_pnl_pct']:+.3f}%"
    )
    eq = report['execution_quality']
    print(
        f"Execution: avg entry improvement {eq['avg_price_improvement_pct']:+.3f}% | "
        f"avg target capture {eq['avg_target_capture_pct']:+.1f}%"
    )
    print('Top model attribution:')
    for name, stats in list(report['by_model'].items())[:5]:
        print(f"  {name}: n={stats['trades']} avg={stats['avg_pnl_pct']:+.3f}% total={stats['total_pnl_pct']:+.3f}%")


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

    # --- Selection diagnostic: is the classifier picking better names than it
    #     discards? (buy&hold by assigned signal, free of execution mechanics) ---
    selection = report.get('selection', {})
    if selection:
        print('\nSelection — buy-and-hold return by assigned signal (mechanics removed):')
        for sig in ('ACTIVE', 'SKIP', 'STALE'):
            s = selection.get(sig)
            if s and s.get('n'):
                print(f"  {sig:<8} n={s['n']:<5} mean buy&hold={s['mean']:+.3f}%/name")
        active_s = selection.get('ACTIVE', {})
        skip_s = selection.get('SKIP', {})
        if active_s.get('n') and skip_s.get('n'):
            diff = active_s['mean'] - skip_s['mean']
            verdict = ('picks BETTER names than it skips' if diff > 0
                       else 'picks WORSE names than it skips — selection is anti-predictive')
            print(f"  -> ACTIVE minus SKIP = {diff:+.3f}%/name: classifier {verdict}.")


def main() -> None:
    parser = argparse.ArgumentParser(description='Backtest OracleForge predictions against realized OHLC.')
    parser.add_argument('--from-date', help='Start date YYYY-MM-DD (inclusive)')
    parser.add_argument('--to-date', help='End date YYYY-MM-DD (inclusive)')
    parser.add_argument('--max-dates', type=int,
                        help='Use only the most recent N prediction dates after date filtering.')
    parser.add_argument('--execution', choices=EXECUTION_MODES, default='limit_stop',
                        help='Execution model to simulate (default: limit_stop = live behavior).')
    parser.add_argument('--compare-executions', action='store_true',
                        help='Run all execution models and print an A/B comparison of the ACTIVE edge.')
    parser.add_argument('--walk-forward', action='store_true',
                        help='Run rolling train/test parameter selection and save walk-forward report.')
    parser.add_argument('--train-window', type=int, default=4,
                        help='Prediction dates per walk-forward training window.')
    parser.add_argument('--test-window', type=int, default=1,
                        help='Prediction dates per walk-forward validation window.')
    parser.add_argument('--attribution', action='store_true',
                        help='Summarise actual closed trade attribution and execution quality.')
    args = parser.parse_args()

    dates = list_prediction_dates()
    if args.from_date:
        dates = [d for d in dates if d >= args.from_date]
    if args.to_date:
        dates = [d for d in dates if d <= args.to_date]
    if args.max_dates and args.max_dates > 0:
        dates = dates[-args.max_dates:]

    if not dates:
        print('No prediction history files found in history/.')
        return

    print(f"Backtesting {len(dates)} day(s): {dates[0]} .. {dates[-1]}")

    if args.compare_executions:
        print("\n=== Execution A/B (ACTIVE signals) ===")
        print(f"{'Execution':<14} {'N':>5} {'Strategy ret%':>14} {'Edge vs hold%':>15} {'Signif?':>9}")
        print('-' * 60)
        comp = compare_executions(dates)
        for mode in EXECUTION_MODES:
            strat = comp[mode]['active_strategy']
            edge = comp[mode]['active_edge_vs_buy_hold']
            flag = 'YES' if edge.get('significant') else ('thin' if not edge.get('sample_adequate') else 'no')
            print(f"{mode:<14} {strat.get('n', 0):>5} {strat.get('mean', 0):>13.3f}% "
                  f"{edge.get('mean', 0):>14.3f}% {flag:>9}")
        print("\n(limit_stop = live; limit_hold = drop stop/target, exit at close; "
              "market_hold = ignore ranges, buy open/sell close)")
        return

    if args.walk_forward:
        report = walk_forward_optimize(
            dates,
            train_window=args.train_window,
            test_window=args.test_window,
        )
        save_json(WALK_FORWARD_FILE, report)
        print_walk_forward_summary(report)
        print(f"\nWalk-forward report saved to {WALK_FORWARD_FILE}")
        return

    if args.attribution:
        journal = load_json('history/trade_journal.json', [])
        report = summarize_trade_attribution(journal)
        save_json(ATTRIBUTION_FILE, report)
        print_attribution_summary(report)
        print(f"\nTrade attribution saved to {ATTRIBUTION_FILE}")
        return

    report = run_backtest(dates, execution=args.execution)

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
