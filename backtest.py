# backtest.py
"""Replay stored signals under the live trader's rules and report the result.

The simulation follows trader.py rather than an idealized fill model: DAY
limit entries at the open, a profit target that only works from the day
after entry, a stop and max hold checked at each close whose market sell
fills at the next open, and one position per ticker. Signals come from the
predictions files as classified at the time, so there is no lookahead from
today's model scores. See simulate_position for the exact rules.
"""
from __future__ import annotations

import argparse
import bisect
import itertools
import math
import os
import statistics
from collections import defaultdict
from datetime import datetime, timedelta

from zoneinfo import ZoneInfo

from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

import alpaca_client
from signals import (
    HISTORY_DIR,
    REPORTS_DIR,
    extract_model_predictions,
    list_prediction_dates,
    load_json,
    save_json,
)

BACKTEST_DIR = 'reports/'
BACKTEST_FILE = os.path.join(BACKTEST_DIR, 'backtest_summary.json')
BACKTEST_HISTORY_DIR = 'reports/backtest_history/'
WALK_FORWARD_FILE = 'reports/walk_forward_summary.json'
ATTRIBUTION_FILE = 'reports/trade_attribution.json'
TRADING_CONFIG_FILE = 'config/trading.json'
JOURNAL_FILE = 'history/trade_journal.json'
BENCHMARK_TICKER = 'SPY'
ET = ZoneInfo('America/New_York')
DEFAULT_ORDER_USD = 2.0
DEFAULT_TRADING_CONFIG = {'stop_loss_pct': 0.95, 'max_hold_days': 15, 'max_per_trade_usd': DEFAULT_ORDER_USD}
MIN_ADEQUATE_SAMPLE = 30   # trades below this -> treat metrics as not yet trustworthy
T_CRIT_95 = 1.96           # ~95% two-sided (normal approx; conservative for small n)
# Execution parameters plus a tighter upside filter: all applicable to the
# stored signals without re-deriving them from model scores.
DEFAULT_WALK_FORWARD_GRID = {
    'stop_loss_pct': [0.93, 0.95, 0.97],
    'max_hold_days': [5, 10, 15],
    'min_upside_pct': [1.5, 2.5, 4.0],
}


def load_trading_config() -> dict:
    cfg = {**DEFAULT_TRADING_CONFIG, **load_json(TRADING_CONFIG_FILE, {})}
    return {k: float(cfg[k]) if k != 'max_hold_days' else int(cfg[k]) for k in DEFAULT_TRADING_CONFIG}


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



# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------

def fetch_daily_bars(tickers: list[str], start: str, end: str | None = None) -> dict[str, dict[str, dict]]:
    """Daily OHLC as {ticker: {date: {open, high, low, close}}} from Alpaca.

    SIP (consolidated) bars, whose highs and lows are what paper fills are
    judged against, unadjusted so they compare directly with the limit
    prices the models produced. Errors propagate: a silently dropped batch
    would bias every number. (yfinance was replaced for exactly that: when
    throttled it reports symbols as "possibly delisted" and returns nothing.)
    Today's bar is left out until after the close, since it is still forming.
    """
    client = alpaca_client.get_data_client()
    now_et = datetime.now(ET)
    # The free data plan refuses SIP queries reaching into the last 15 minutes.
    latest = now_et - timedelta(minutes=20)
    end_dt = min(datetime.strptime(end, '%Y-%m-%d').replace(tzinfo=ET), latest) if end else latest
    # Some stored tickers use Yahoo's class-share spelling (BRK-B); Alpaca
    # wants BRK.B and rejects the whole batch over one invalid symbol.
    names = {t.replace('-', '.'): t for t in set(tickers) | {BENCHMARK_TICKER}}
    symbols = sorted(names)
    today = now_et.strftime('%Y-%m-%d')
    drop_today = now_et.hour < 16 or (now_et.hour == 16 and now_et.minute < 15)
    out: dict[str, dict[str, dict]] = {}
    for i in range(0, len(symbols), 200):
        req = StockBarsRequest(
            symbol_or_symbols=symbols[i:i + 200],
            timeframe=TimeFrame.Day,
            start=datetime.strptime(start, '%Y-%m-%d').replace(tzinfo=ET),
            end=end_dt,
            feed=DataFeed.SIP,
            adjustment=Adjustment.RAW,
        )
        for sym, sym_bars in client.get_stock_bars(req).data.items():
            series = {}
            for bar in sym_bars:
                date = bar.timestamp.astimezone(ET).strftime('%Y-%m-%d')
                if date == today and drop_today:
                    continue
                series[date] = {'open': float(bar.open), 'high': float(bar.high),
                                'low': float(bar.low), 'close': float(bar.close)}
            if series:
                out[names.get(sym, sym)] = series
    return out


def trading_sessions(bars: dict[str, dict[str, dict]]) -> list[str]:
    """Session dates, taken from the benchmark (falls back to all tickers)."""
    if bars.get(BENCHMARK_TICKER):
        return sorted(bars[BENCHMARK_TICKER])
    return sorted({d for series in bars.values() for d in series})


# ---------------------------------------------------------------------------
# Signal days: which predictions file the morning job traded on each session
# ---------------------------------------------------------------------------

def build_signal_days(dates: list[str], sessions: list[str]) -> list[dict]:
    """Map each predictions file to the session its buys were placed in.

    A file dated D is normally traded at the first session after D: files
    are dated by the session they were forecast from (older ones by the
    calendar evening they ran, e.g. a Sunday). Before the session-date fix,
    a forge invocation that started after midnight ET dated its files by
    that new day, and that same morning's job traded them; those show up as
    a signals report generated early on D itself, and map to session D.
    If several files map to one session, the morning job used the newest.
    """
    by_session: dict[str, str] = {}
    for d in dates:
        generated = str(load_json(os.path.join(REPORTS_DIR, f'signals_{d}.json'), {}).get('generated_at', ''))
        # generated_at is the runner's local clock; before 09:00 is before
        # the open in both runner timezones (UTC and US Central).
        same_morning = generated[:10] == d and generated[11:13] < '09'
        i = (bisect.bisect_left if same_morning else bisect.bisect_right)(sessions, d)
        if i < len(sessions):
            s = sessions[i]
            by_session[s] = max(d, by_session.get(s, d))
    days = []
    for session, signal_date in sorted(by_session.items()):
        predictions = load_json(os.path.join(HISTORY_DIR, f'predictions_{signal_date}.json'), {})
        days.append({'signal_date': signal_date, 'entry_session': session, 'predictions': predictions})
    return days


def consensus_source(min_upside_pct: float | None = None):
    """Ranges the live trader bought: stored ACTIVE signals, as classified at
    the time (no recomputation with today's model scores). An optional
    min_upside_pct can only tighten that set."""
    def source(entry: dict):
        if not isinstance(entry, dict) or entry.get('signal') != 'ACTIVE':
            return None
        if min_upside_pct is not None and float(entry.get('upside_pct') or 0) < min_upside_pct:
            return None
        return entry.get('consensus')
    return source


def model_source(model_name: str):
    """Every non-fallback range one model produced, traded on its own."""
    def source(entry: dict):
        return extract_model_predictions(entry).get(model_name)
    return source


# ---------------------------------------------------------------------------
# Live-rule simulation
# ---------------------------------------------------------------------------

def simulate_position(bars: dict[str, dict], sessions: list[str], entry_idx: int,
                      signal_date: str, buy_high: float, sell_low: float,
                      stop_loss_pct: float, max_hold_days: int,
                      end_idx: int | None = None) -> dict | None:
    """Replay one ACTIVE signal under trader.py's rules. None if not filled.

    - Entry: DAY limit buy at buy_high placed at the open of the entry
      session. Fills at the open if it gaps below buy_high, else at buy_high
      if the session trades down to it.
    - Target: DAY limit sell at sell_low, placed after the entry day's close,
      so it works from the next session on; fills at the open on a gap up.
    - Stop / max hold: checked against each close (entry day included). The
      market sell goes in after the close, so it fills at the next open;
      booked_exit_price is the close the live journal records instead.
      Max hold counts calendar days from the signal date, like trader.py.
    - Still held at end_idx: exit_reason 'open', valued at the last close.
    """
    end_idx = len(sessions) - 1 if end_idx is None else end_idx
    day = bars.get(sessions[entry_idx])
    if day is None or day['low'] > buy_high:
        return None
    entry_price = min(day['open'], buy_high)
    stop_level = entry_price * stop_loss_pct
    signal_dt = datetime.strptime(signal_date, '%Y-%m-%d').date()

    def trade(exit_idx, exit_price, reason, booked=None, booked_idx=None):
        exit_price = round(exit_price, 4)
        booked = round(booked if booked is not None else exit_price, 4)
        return {
            'signal_date': signal_date,
            'entry_date': sessions[entry_idx],
            'entry_price': round(entry_price, 4),
            'buy_high': buy_high,
            'sell_low': sell_low,
            'exit_date': sessions[exit_idx],
            'exit_price': exit_price,
            'booked_exit_date': sessions[booked_idx if booked_idx is not None else exit_idx],
            'booked_exit_price': booked,
            'exit_reason': reason,
            'return_pct': round((exit_price / entry_price - 1) * 100, 4),
            'booked_return_pct': round((booked / entry_price - 1) * 100, 4),
            'sessions_held': exit_idx - entry_idx,
        }

    pending_exit = None  # (reason, booked close, booked idx) from last evening
    last_idx = entry_idx
    for i in range(entry_idx, end_idx + 1):
        bar = bars.get(sessions[i])
        if bar is None:
            continue  # no print that session (halt/missing data)
        last_idx = i
        if pending_exit:
            reason, booked, booked_idx = pending_exit
            return trade(i, bar['open'], reason, booked, booked_idx)
        if i > entry_idx and bar['high'] >= sell_low:
            return trade(i, max(bar['open'], sell_low), 'target')
        held_days = (datetime.strptime(sessions[i], '%Y-%m-%d').date() - signal_dt).days
        if bar['close'] <= stop_level:
            pending_exit = ('stop', bar['close'], i)
        elif held_days >= max_hold_days:
            pending_exit = ('max_hold', bar['close'], i)
    last_close = bars[sessions[last_idx]]['close']
    return trade(last_idx, last_close, 'open')


def simulate_strategy(signal_days: list[dict], bars: dict[str, dict[str, dict]],
                      sessions: list[str], source, stop_loss_pct: float,
                      max_hold_days: int, end_idx: int | None = None) -> dict:
    """Walk signal days in order with one position per ticker, like the live
    trader (a new signal for a held ticker is skipped). A ticker frees up at
    its exit session for stop/max-hold exits (tracking was cleared the
    evening before) and the session after a target fill."""
    session_idx = {s: i for i, s in enumerate(sessions)}
    end_idx = len(sessions) - 1 if end_idx is None else end_idx
    free_from: dict[str, int] = {}
    trades, unfilled, skipped_held, missing = [], 0, 0, 0
    for day in signal_days:
        i = session_idx[day['entry_session']]
        if i > end_idx:
            break
        for ticker, entry in day['predictions'].items():
            rng = source(entry)
            if not rng:
                continue
            try:
                buy_high = float(rng['buy_high'])
                sell_low = float(rng['sell_low'])
            except (KeyError, TypeError, ValueError):
                continue
            if buy_high <= 0 or sell_low <= buy_high:
                continue
            if free_from.get(ticker, 0) > i:
                skipped_held += 1
                continue
            series = bars.get(ticker)
            if not series or day['entry_session'] not in series:
                missing += 1
                continue
            t = simulate_position(series, sessions, i, day['signal_date'], buy_high, sell_low,
                                  stop_loss_pct, max_hold_days, end_idx)
            if t is None:
                unfilled += 1
                continue
            t['ticker'] = ticker
            t['hold_return_pct'] = _hold_return(series, t)
            trades.append(t)
            exit_i = session_idx[t['exit_date']]
            free_from[ticker] = (len(sessions) + 1 if t['exit_reason'] == 'open'
                                 else exit_i + 1 if t['exit_reason'] == 'target' else exit_i)
    return {'trades': trades, 'unfilled': unfilled, 'skipped_already_held': skipped_held,
            'missing_bars': missing}


def _hold_return(series: dict[str, dict], t: dict) -> float:
    """Buy-and-hold over the trade's own window: entry-session open to the
    exit session's close. Same name, same days, no limits or stops."""
    start = series[t['entry_date']]['open']
    end = series[t['exit_date']]['close']
    return round((end / start - 1) * 100, 4) if start else 0.0


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summarize_trades(trades: list[dict], order_usd: float = DEFAULT_ORDER_USD,
                     key: str = 'return_pct') -> dict:
    """Closed-trade stats; positions still open are counted but not scored."""
    closed = [t for t in trades if t['exit_reason'] != 'open']
    rets = [t[key] for t in closed]
    wins = [r for r in rets if r > 0]
    losses = [-r for r in rets if r < 0]
    streak = max_streak = 0
    for r in rets:
        streak = streak + 1 if r < 0 else (0 if r > 0 else streak)
        max_streak = max(max_streak, streak)
    reasons = defaultdict(list)
    for t in closed:
        reasons[t['exit_reason']].append(t[key])
    return {
        'trades': len(closed),
        'open_positions': len(trades) - len(closed),
        'win_rate': round(len(wins) / len(rets), 4) if rets else 0.0,
        'avg_return_pct': _avg(rets),
        'avg_win_pct': _avg(wins),
        'avg_loss_pct': _avg(losses),
        'profit_factor': round(sum(wins) / sum(losses), 3) if losses else None,
        'max_consecutive_losses': max_streak,
        'avg_sessions_held': _avg([t['sessions_held'] for t in closed]),
        'total_pnl_usd': round(sum(order_usd * r / 100 for r in rets), 2),
        'by_exit_reason': {
            reason: {'trades': len(v), 'avg_return_pct': _avg(v),
                     'total_pnl_usd': round(sum(order_usd * r / 100 for r in v), 2)}
            for reason, v in sorted(reasons.items())
        },
        'significance': _stats(rets),
    }


def calibrate_against_journal(trades: list[dict], journal: list[dict]) -> dict:
    """Match simulated trades to live ones on (ticker, signal date).

    The live journal records the signal date as entry_date. Live exits are
    booked at the close for stop/max-hold, so compare booked returns. Trades
    flagged qty_desync_corrected are excluded: their P&L reflects a share-
    count bug, not the strategy.
    """
    first = min((t['signal_date'] for t in trades), default=None)
    last = max((t['signal_date'] for t in trades), default=None)
    live = {
        (t.get('ticker'), t.get('entry_date')): t for t in journal
        if not t.get('qty_desync_corrected') and first and first <= (t.get('entry_date') or '') <= last
    }
    sim = {(t['ticker'], t['signal_date']): t for t in trades if t['exit_reason'] != 'open'}
    matched = sorted(set(live) & set(sim))
    diffs = [sim[k]['booked_return_pct'] - float(live[k].get('pnl_pct') or 0) for k in matched]
    same_side = sum(
        1 for k in matched
        if (sim[k]['booked_return_pct'] > 0) == (float(live[k].get('pnl_pct') or 0) > 0)
    )
    return {
        'live_trades_in_window': len(live),
        'simulated_trades_in_window': len(sim),
        'matched': len(matched),
        'live_only': len(set(live) - set(sim)),
        'simulated_only': len(set(sim) - set(live)),
        'same_win_loss': same_side,
        'agreement_rate': round(same_side / len(matched), 4) if matched else None,
        'avg_live_return_pct': _avg([float(live[k].get('pnl_pct') or 0) for k in matched]),
        'avg_sim_booked_return_pct': _avg([sim[k]['booked_return_pct'] for k in matched]),
        'mean_abs_return_gap_pct': _avg([abs(d) for d in diffs]),
    }


def selection_diagnostic(signal_days: list[dict], bars: dict[str, dict[str, dict]]) -> dict:
    """Entry-session open-to-close return by stored signal, free of any limit
    or stop mechanics: does the classifier pick better names than it skips?"""
    by_signal: dict[str, list] = defaultdict(list)
    for day in signal_days:
        for ticker, entry in day['predictions'].items():
            bar = (bars.get(ticker) or {}).get(day['entry_session'])
            if not bar or not isinstance(entry, dict) or not bar['open']:
                continue
            by_signal[entry.get('signal', 'SKIP')].append((bar['close'] / bar['open'] - 1) * 100)
    return {sig: _stats(v) for sig, v in sorted(by_signal.items())}


def _all_tickers(signal_days: list[dict]) -> list[str]:
    return sorted({t for day in signal_days for t in day['predictions']})


def load_market(dates: list[str]) -> tuple[list[dict], dict, list[str]]:
    """Download bars once and map prediction files to entry sessions."""
    if not dates:
        return [], {}, []
    tickers = sorted({
        t for d in dates
        for t in load_json(os.path.join(HISTORY_DIR, f'predictions_{d}.json'), {})
    })
    bars = fetch_daily_bars(tickers, start=dates[0])
    sessions = trading_sessions(bars)
    return build_signal_days(dates, sessions), bars, sessions


def run_backtest(dates: list[str], stop_loss_pct: float | None = None,
                 max_hold_days: int | None = None, market=None) -> dict:
    cfg = load_trading_config()
    stop_loss_pct = cfg['stop_loss_pct'] if stop_loss_pct is None else stop_loss_pct
    max_hold_days = cfg['max_hold_days'] if max_hold_days is None else max_hold_days
    order_usd = cfg['max_per_trade_usd']
    signal_days, bars, sessions = market or load_market(dates)

    result = simulate_strategy(signal_days, bars, sessions, consensus_source(),
                               stop_loss_pct, max_hold_days)
    trades = result['trades']
    closed = [t for t in trades if t['exit_reason'] != 'open']
    edge = [t['return_pct'] - t['hold_return_pct'] for t in closed]

    models = sorted({m for day in signal_days for e in day['predictions'].values()
                     for m in extract_model_predictions(e)})
    by_model = {
        m: summarize_trades(simulate_strategy(signal_days, bars, sessions, model_source(m),
                                              stop_loss_pct, max_hold_days)['trades'], order_usd)
        for m in models
    }

    return {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'method': 'live-rules replay of stored ACTIVE signals',
        'params': {'stop_loss_pct': stop_loss_pct, 'max_hold_days': max_hold_days,
                   'order_usd': order_usd, 'stop_fill': 'next session open'},
        'signal_dates': [d['signal_date'] for d in signal_days],
        'days_in_history': len(signal_days),
        'last_session': sessions[-1] if sessions else None,
        'strategy': summarize_trades(trades, order_usd),
        'strategy_as_booked': summarize_trades(trades, order_usd, key='booked_return_pct'),
        'orders': {'filled': len(trades), 'unfilled': result['unfilled'],
                   'skipped_already_held': result['skipped_already_held'],
                   'missing_bars': result['missing_bars']},
        'benchmark': {
            'strategy': _stats([t['return_pct'] for t in closed]),
            'buy_hold_same_window': _stats([t['hold_return_pct'] for t in closed]),
            'edge_vs_buy_hold': _stats(edge),
        },
        'by_model': by_model,
        'selection': selection_diagnostic(signal_days, bars),
        'calibration': calibrate_against_journal(trades, load_json(JOURNAL_FILE, [])),
        'trades': trades,
    }


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------

def _grid_candidates(grid: dict | None = None) -> list[dict]:
    """Expand a small parameter grid into concrete candidate configs."""
    grid = grid or DEFAULT_WALK_FORWARD_GRID
    keys = list(grid.keys())
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def _evaluate(signal_days, bars, sessions, candidate, end_idx=None) -> dict:
    result = simulate_strategy(
        signal_days, bars, sessions, consensus_source(candidate.get('min_upside_pct')),
        candidate['stop_loss_pct'], candidate['max_hold_days'], end_idx,
    )
    return _stats([t['return_pct'] for t in result['trades'] if t['exit_reason'] != 'open'])


def walk_forward_optimize(signal_days: list[dict], bars: dict, sessions: list[str],
                          train_window: int = 20, test_window: int = 5,
                          grid: dict | None = None, min_train_trades: int = MIN_ADEQUATE_SAMPLE) -> dict:
    """Pick params on a trailing window of signal days, validate on the next.

    Training outcomes are cut off before the test window starts, so a pick
    never sees prices from the period it is judged on. Windows with too few
    training trades to rank, or no test trades, are reported but excluded
    from the averages (an empty window is missing evidence, not a 0% edge).
    Never mutates live config.
    """
    candidates = _grid_candidates(grid)
    session_idx = {s: i for i, s in enumerate(sessions)}
    windows = []
    for end in range(train_window, len(signal_days), test_window):
        train = signal_days[end - train_window:end]
        test = signal_days[end:end + test_window]
        cutoff = session_idx[test[0]['entry_session']] - 1
        ranked = sorted(
            ((_evaluate(train, bars, sessions, c, cutoff), c) for c in candidates),
            key=lambda row: (row[0]['n'] >= min_train_trades, row[0]['mean'], row[0]['n']),
            reverse=True,
        )
        train_stats, best = ranked[0]
        rankable = train_stats['n'] >= min_train_trades
        test_stats = _evaluate(test, bars, sessions, best) if rankable else None
        windows.append({
            'train': [train[0]['signal_date'], train[-1]['signal_date']],
            'test': [test[0]['signal_date'], test[-1]['signal_date']],
            'selected_params': best if rankable else None,
            'train_stats': train_stats,
            'test_stats': test_stats,
        })
    scored = [w for w in windows if w['test_stats'] and w['test_stats']['n'] > 0]
    test_means = [w['test_stats']['mean'] for w in scored]

    latest = None
    if len(signal_days) >= train_window:
        recent = signal_days[-train_window:]
        stats, best = max(((_evaluate(recent, bars, sessions, c), c) for c in candidates),
                          key=lambda row: (row[0]['n'] >= min_train_trades, row[0]['mean'], row[0]['n']))
        latest = {'train': [recent[0]['signal_date'], recent[-1]['signal_date']],
                  'selected_params': best if stats['n'] >= min_train_trades else None,
                  'train_stats': stats}

    return {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'train_window': train_window,
        'test_window': test_window,
        'min_train_trades': min_train_trades,
        'candidate_count': len(candidates),
        'windows': windows,
        'summary': {
            'windows': len(windows),
            'validation_windows': len(scored),
            'validation_trades': sum(w['test_stats']['n'] for w in scored),
            'avg_validation_return_pct': _avg(test_means),
            'positive_windows': sum(1 for m in test_means if m > 0),
        },
        'latest_recommendation': latest,
    }


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def _fmt_stats_line(label: str, s: dict) -> str:
    if not s or not s.get('n'):
        return f"  {label:<24} (no data)"
    adq = 'ok' if s['sample_adequate'] else f'THIN (<{MIN_ADEQUATE_SAMPLE})'
    return (f"  {label:<24} n={s['n']:<5} mean={s['mean']:+.3f}%  "
            f"95%CI [{s['ci95_low']:+.3f}, {s['ci95_high']:+.3f}]  sample={adq}")


def print_backtest_summary(report: dict) -> None:
    p = report['params']
    print('\n=== OracleForge Backtest (live-rules replay) ===')
    print(f"Signal days: {report['days_in_history']} | stop {p['stop_loss_pct']} | "
          f"max hold {p['max_hold_days']}d | ${p['order_usd']:.2f}/order | stop fills at {p['stop_fill']}")
    o = report['orders']
    print(f"Orders: {o['filled']} filled, {o['unfilled']} unfilled, "
          f"{o['skipped_already_held']} skipped (already held), {o['missing_bars']} no data")

    for label, key in (('Strategy (realistic fills)', 'strategy'),
                       ('Strategy (as the live journal books it)', 'strategy_as_booked')):
        s = report[key]
        pf = f"{s['profit_factor']:.2f}" if s['profit_factor'] is not None else 'N/A'
        print(f"\n{label}: {s['trades']} closed, {s['open_positions']} open | "
              f"win {s['win_rate'] * 100:.1f}% | avg {s['avg_return_pct']:+.3f}% | PF {pf} | "
              f"P&L ${s['total_pnl_usd']:+.2f}")
        for reason, r in s['by_exit_reason'].items():
            print(f"    {reason:<9} n={r['trades']:<5} avg {r['avg_return_pct']:+.3f}%  ${r['total_pnl_usd']:+.2f}")

    b = report['benchmark']
    print('\nBenchmark: each trade vs holding the same name over the same sessions')
    print(_fmt_stats_line('Strategy', b['strategy']))
    print(_fmt_stats_line('Buy & hold', b['buy_hold_same_window']))
    print(_fmt_stats_line('Edge', b['edge_vs_buy_hold']))

    print('\nBy model (each model\'s own ranges, same rules):')
    for model, s in report['by_model'].items():
        print(f"  {model:<32} n={s['trades']:<5} win {s['win_rate'] * 100:5.1f}%  avg {s['avg_return_pct']:+.3f}%")

    sel = report['selection']
    if sel.get('ACTIVE', {}).get('n') and sel.get('SKIP', {}).get('n'):
        diff = sel['ACTIVE']['mean'] - sel['SKIP']['mean']
        print(f"\nSelection: entry-session open-to-close, ACTIVE {sel['ACTIVE']['mean']:+.3f}% vs "
              f"SKIP {sel['SKIP']['mean']:+.3f}% ({diff:+.3f}%/name)")

    c = report['calibration']
    if c['matched']:
        print(f"\nCalibration vs live journal: {c['matched']} matched trades, same win/loss on "
              f"{c['agreement_rate'] * 100:.0f}%; avg return live {c['avg_live_return_pct']:+.3f}% vs "
              f"simulated {c['avg_sim_booked_return_pct']:+.3f}%; {c['live_only']} live-only, "
              f"{c['simulated_only']} simulated-only")


def print_walk_forward_summary(report: dict) -> None:
    s = report['summary']
    print('\n=== Walk-Forward ===')
    print(f"Windows: {s['windows']} ({s['validation_windows']} with validation trades, "
          f"{s['validation_trades']} trades) | avg validation return {s['avg_validation_return_pct']:+.3f}% | "
          f"positive {s['positive_windows']}")
    rec = report.get('latest_recommendation')
    if rec and rec['selected_params']:
        print(f"Latest pick on {rec['train'][0]}..{rec['train'][1]} "
              f"(n={rec['train_stats']['n']}, mean {rec['train_stats']['mean']:+.3f}%): {rec['selected_params']}")
    elif rec:
        print(f"Latest window has {rec['train_stats']['n']} trades, below {report['min_train_trades']}: no pick.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description='Replay OracleForge signals under the live trading rules.')
    parser.add_argument('--from-date', help='First predictions date YYYY-MM-DD (inclusive)')
    parser.add_argument('--to-date', help='Last predictions date YYYY-MM-DD (inclusive)')
    parser.add_argument('--max-dates', type=int, help='Use only the most recent N predictions files.')
    parser.add_argument('--stop-loss-pct', type=float, help='Override config stop_loss_pct.')
    parser.add_argument('--max-hold-days', type=int, help='Override config max_hold_days.')
    parser.add_argument('--walk-forward', action='store_true',
                        help='Also run the rolling train/test study and save its report.')
    parser.add_argument('--train-window', type=int, default=20, help='Signal days per training window.')
    parser.add_argument('--test-window', type=int, default=5, help='Signal days per validation window.')
    parser.add_argument('--attribution', action='store_true',
                        help='Summarise actual closed trade attribution and execution quality.')
    args = parser.parse_args()

    if args.attribution:
        report = summarize_trade_attribution(load_json(JOURNAL_FILE, []))
        save_json(ATTRIBUTION_FILE, report)
        print_attribution_summary(report)
        print(f"\nTrade attribution saved to {ATTRIBUTION_FILE}")
        return

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

    print(f"Backtesting {len(dates)} predictions file(s): {dates[0]} .. {dates[-1]}")
    market = load_market(dates)
    if not market[2]:
        print('No market data downloaded; nothing to simulate.')
        return

    report = run_backtest(dates, args.stop_loss_pct, args.max_hold_days, market=market)
    save_json(BACKTEST_FILE, report)
    os.makedirs(BACKTEST_HISTORY_DIR, exist_ok=True)
    archive = os.path.join(BACKTEST_HISTORY_DIR, f"backtest_{datetime.now():%Y-%m-%d}.json")
    save_json(archive, {k: v for k, v in report.items() if k != 'trades'})
    print_backtest_summary(report)
    print(f"\nReport saved to {BACKTEST_FILE} (snapshot {archive})")

    if args.walk_forward:
        wf = walk_forward_optimize(*market, train_window=args.train_window, test_window=args.test_window)
        save_json(WALK_FORWARD_FILE, wf)
        print_walk_forward_summary(wf)
        print(f"Walk-forward report saved to {WALK_FORWARD_FILE}")


if __name__ == '__main__':
    main()
