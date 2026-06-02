# forge_loop.py — overnight inference engine
import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import requests
import yfinance as yf
from alpaca.data.enums import DataFeed
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

import alpaca_client
from signals import (
    build_enriched_predictions,
    build_signals_report,
    extract_model_predictions,
    parse_ticker_list,
    print_signals_table,
    save_json as save_signals_json,
)

CONFIG_FILE = 'config/tickers.json'
MODELS_FILE = 'config/models.json'
SCORES_FILE = 'state/analyst_scores.json'
HISTORY_DIR = 'history/'
REPORTS_DIR = 'reports/'
TRADE_JOURNAL_FILE = 'history/trade_journal.json'
MAX_PREDICTION_LOOKBACK_DAYS = 10
MAX_LLM_RETRIES = 3
LLM_RETRY_DELAY = 5   # seconds between retry attempts
EARNINGS_LOOKAHEAD_DAYS = 2

os.makedirs('config', exist_ok=True)
os.makedirs('state', exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def load_json(filepath, default_data):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return default_data


def save_json(filepath, data):
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)


def find_latest_predictions_path():
    for days_back in range(1, MAX_PREDICTION_LOOKBACK_DAYS + 1):
        date_str = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        path = os.path.join(HISTORY_DIR, f'predictions_{date_str}.json')
        if os.path.exists(path):
            return path, date_str
    return None, None


def _headline_from_news_item(item):
    if not isinstance(item, dict):
        return None
    if 'title' in item:
        title = item['title']
        publisher = item.get('publisher') or item.get('provider', 'Unknown')
        if isinstance(publisher, dict):
            publisher = publisher.get('displayName', 'Unknown')
        return title, publisher
    content = item.get('content')
    if isinstance(content, dict) and content.get('title'):
        title = content['title']
        provider = content.get('provider') or {}
        publisher = (
            provider.get('displayName', 'Unknown')
            if isinstance(provider, dict)
            else 'Unknown'
        )
        return title, publisher
    return None


def fetch_recent_news(ticker):
    try:
        stock = yf.Ticker(ticker)
        news_items = stock.news
        if not news_items:
            return "No recent news available."
        headlines = []
        for item in news_items[:5]:
            parsed = _headline_from_news_item(item)
            if parsed:
                title, publisher = parsed
                headlines.append(f"- {title} ({publisher})")
        return "\n".join(headlines) if headlines else "No recent news available."
    except Exception:
        return "No recent news available."


def has_upcoming_earnings(ticker: str) -> bool:
    """Return True if ticker has earnings within EARNINGS_LOOKAHEAD_DAYS trading days."""
    try:
        from datetime import date as date_type
        stock = yf.Ticker(ticker)
        cal = stock.calendar
        if not cal:
            return False
        # yfinance returns a dict in newer versions
        if isinstance(cal, dict):
            raw_dates = cal.get('Earnings Date', [])
            if not isinstance(raw_dates, list):
                raw_dates = [raw_dates]
        else:
            # older DataFrame format
            raw_dates = cal.loc['Earnings Date'].tolist() if 'Earnings Date' in cal.index else []
        today = datetime.now().date()
        cutoff = today + timedelta(days=EARNINGS_LOOKAHEAD_DAYS)
        for d in raw_dates:
            if hasattr(d, 'date'):
                d = d.date()
            elif isinstance(d, str):
                d = datetime.strptime(d[:10], '%Y-%m-%d').date()
            if isinstance(d, date_type) and today <= d <= cutoff:
                return True
        return False
    except Exception:
        return False


def _fallback_range(price: float) -> dict:
    """Synthetic fallback when LLM output is unparseable. Tagged so it can be excluded."""
    return {
        'buy_low': round(price * 0.985, 2),
        'buy_high': round(price * 0.995, 2),
        'sell_low': round(price * 1.015, 2),
        'sell_high': round(price * 1.025, 2),
        'rationale': 'Fallback range (model output unparseable)',
        'fallback': True,
    }


def parse_llm_range(raw_output: str, fallback_price: float) -> dict:
    if not raw_output or not raw_output.strip():
        return _fallback_range(fallback_price)

    text = re.sub(r'```(?:json)?\s*', '', raw_output.strip()).strip('`').strip()

    def _try_parse(s: str) -> dict | None:
        try:
            data = json.loads(s)
            bl = float(data.get('buy_low', 0))
            bh = float(data.get('buy_high', 0))
            sl = float(data.get('sell_low', 0))
            sh = float(data.get('sell_high', 0))
            if bl > 0 and bh > bl and sl > bh and sh >= sl:
                return {
                    'buy_low': round(bl, 2),
                    'buy_high': round(bh, 2),
                    'sell_low': round(sl, 2),
                    'sell_high': round(sh, 2),
                    'rationale': str(data.get('rationale', ''))[:200],
                }
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        return None

    result = _try_parse(text)
    if result:
        return result
    m = re.search(r'\{[^{}]+\}', text, re.DOTALL)
    if m:
        result = _try_parse(m.group())
        if result:
            return result
    return _fallback_range(fallback_price)


def evaluate_range_prediction(high_price, low_price, pred, stop_threshold: float = 0.95) -> float:
    """Score a prior range prediction against realized OHLC: +0.01, -0.01, or 0.0."""
    if not isinstance(pred, dict):
        return 0.0
    # Fallback predictions are synthetic — do not reward or penalise them
    if pred.get('fallback'):
        return 0.0
    buy_high = float(pred.get('buy_high') or 0)
    sell_low = float(pred.get('sell_low') or 0)
    if buy_high <= 0 or sell_low <= 0:
        return 0.0
    if low_price > buy_high:
        return 0.0
    stop_limit = buy_high * stop_threshold
    if low_price <= stop_limit:
        return -0.01
    if high_price >= sell_low:
        return 0.01
    return 0.0  # price entered range but trade still open — no penalty


def score_deltas_from_journal(journal: list, prior_date: str, scores: dict) -> dict:
    """
    Build score deltas from actual closed trades on prior_date.
    Models that predicted a winning trade get +0.02; losers get -0.02.
    Weighted higher than theoretical OHLC evaluation (+-0.01) to reflect real outcomes.
    """
    deltas: dict[str, list] = defaultdict(list)
    for trade in journal:
        if trade.get('close_date') != prior_date:
            continue
        delta = 0.02 if trade.get('outcome') == 'win' else -0.02
        for model in trade.get('predicting_models', []):
            if model in scores:
                deltas[model].append(delta)
    return deltas


def apply_score_deltas(scores: dict, deltas_by_model: dict, decay: float = 1.0) -> None:
    """Apply score adjustments with optional exponential decay.

    decay < 1.0 gradually down-weights stale historical accuracy so recent
    performance carries more influence. decay=1.0 (default) disables decay.
    """
    for model_name, deltas in deltas_by_model.items():
        non_zero = [d for d in deltas if d != 0.0]
        if not non_zero:
            continue
        current = scores.get(model_name, 5.0)
        adjustment = sum(non_zero) / len(non_zero)
        # Apply decay before adding the new adjustment
        new_score = current * decay + adjustment
        scores[model_name] = round(min(10.0, max(0.0, new_score)), 3)


def model_performance_context(model_name: str, journal: list) -> str:
    """Summarise this model's recent actual trade performance for the prompt."""
    trades = [t for t in journal if model_name in t.get('predicting_models', [])]
    if not trades:
        return ""
    recent = trades[-20:]
    wins = sum(1 for t in recent if t['outcome'] == 'win')
    win_rate = wins / len(recent) * 100
    avg_pnl = sum(t['pnl_pct'] for t in recent) / len(recent)
    return (
        f"Your recent trading performance: {win_rate:.0f}% win rate "
        f"over your last {len(recent)} closed trades, "
        f"average P&L {avg_pnl:+.2f}% per trade."
    )


def ticker_trade_history(ticker: str, journal: list, limit: int = 5) -> str:
    """Summarise recent actual trade outcomes for this ticker for the prompt."""
    trades = [t for t in journal if t['ticker'] == ticker][-limit:]
    if not trades:
        return ""
    lines = [f"Recent actual trade results for {ticker}:"]
    for t in trades:
        lines.append(
            f"  {t['close_date']}: bought ${t['entry_price']:.2f} -> "
            f"sold ${t['exit_price']:.2f}  {t['pnl_pct']:+.2f}% ({t['outcome']})"
        )
    return "\n".join(lines)


def compute_technicals(bars: list) -> dict:
    """Compute RSI-14, Bollinger %B, volume ratio, and price context from bar history."""
    if len(bars) < 5:
        return {}
    closes = [float(b.close) for b in bars]
    volumes = [float(b.volume) for b in bars]
    highs = [float(b.high) for b in bars]
    lows = [float(b.low) for b in bars]
    close = closes[-1]

    result = {}

    # RSI-14 (Wilder's smoothed average — requires at least 15 bars)
    if len(closes) >= 15:
        deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        # Seed with simple average of first 14 periods
        seed_gains = [d for d in deltas[:14] if d > 0]
        seed_losses = [-d for d in deltas[:14] if d < 0]
        avg_gain = sum(seed_gains) / 14
        avg_loss = sum(seed_losses) / 14
        # Apply Wilder's smoothing for the remaining bars
        for d in deltas[14:]:
            gain = d if d > 0 else 0.0
            loss = -d if d < 0 else 0.0
            avg_gain = (avg_gain * 13 + gain) / 14
            avg_loss = (avg_loss * 13 + loss) / 14
        result['rsi14'] = 100.0 if avg_loss == 0 else round(100 - (100 / (1 + avg_gain / avg_loss)), 1)

    # Volume ratio vs prior 20-day average (exclude today)
    prior_vols = volumes[-21:-1] if len(volumes) > 20 else volumes[:-1]
    avg_vol = sum(prior_vols) / len(prior_vols) if prior_vols else volumes[-1]
    result['vol_ratio'] = round(volumes[-1] / avg_vol, 2) if avg_vol else 1.0

    # 10-day high/low distance
    n = min(10, len(bars))
    high_10d = max(highs[-n:])
    low_10d = min(lows[-n:])
    result['pct_from_10d_high'] = round((close - high_10d) / high_10d * 100, 1)
    result['pct_from_10d_low'] = round((close - low_10d) / low_10d * 100, 1)

    # SMA-20 distance
    sma_window = closes[-20:] if len(closes) >= 20 else closes
    sma20 = sum(sma_window) / len(sma_window)
    result['pct_from_sma20'] = round((close - sma20) / sma20 * 100, 1)

    # Bollinger Band %B — where is price within the 20-day band?
    if len(closes) >= 20:
        bb_closes = closes[-20:]
        bb_mean = sum(bb_closes) / 20
        bb_std = (sum((c - bb_mean) ** 2 for c in bb_closes) / 20) ** 0.5
        if bb_std > 0:
            upper_band = bb_mean + 2 * bb_std
            lower_band = bb_mean - 2 * bb_std
            band_width = upper_band - lower_band
            result['bb_pct'] = round((close - lower_band) / band_width, 3) if band_width > 0 else 0.5

    # 5-day price momentum
    if len(closes) >= 6:
        result['price_momentum_5d'] = round((close / closes[-6] - 1) * 100, 2)

    return result


def _technicals_block(t: dict) -> str:
    """Format technicals dict as a prompt-ready text block."""
    if not t:
        return ""
    lines = ["Technical context (use to calibrate your price levels):"]
    if t.get('rsi14') is not None:
        rsi = t['rsi14']
        note = ' (oversold — potential bounce)' if rsi < 35 else ' (overbought — caution)' if rsi > 65 else ''
        lines.append(f"- RSI(14): {rsi}{note}")
    if t.get('vol_ratio') is not None:
        vr = t['vol_ratio']
        note = ' (elevated — strong interest)' if vr > 1.5 else ' (below average)' if vr < 0.7 else ''
        lines.append(f"- Volume: {vr}x 20-day average{note}")
    if t.get('pct_from_10d_low') is not None:
        lines.append(
            f"- Price is {t['pct_from_10d_low']:+.1f}% from 10-day low (support) "
            f"and {t['pct_from_10d_high']:+.1f}% from 10-day high (resistance)"
        )
    if t.get('pct_from_sma20') is not None:
        lines.append(f"- Price is {t['pct_from_sma20']:+.1f}% from 20-day SMA")
    if t.get('bb_pct') is not None:
        bp = t['bb_pct']
        note = ' (near upper band — overbought)' if bp > 0.8 else ' (near lower band — oversold)' if bp < 0.2 else ''
        lines.append(f"- Bollinger %B: {bp:.2f}{note}")
    if t.get('price_momentum_5d') is not None:
        lines.append(f"- 5-day momentum: {t['price_momentum_5d']:+.2f}%")
    return "\n".join(lines)


def call_local_llm(ticker: str, current_price: float, model_name: str,
                   journal: list | None = None,
                   technicals: dict | None = None) -> dict:
    journal = journal or []
    news_context = fetch_recent_news(ticker)
    perf_context = model_performance_context(model_name, journal)
    history_context = ticker_trade_history(ticker, journal)
    tech_context = _technicals_block(technicals or {})

    context_block = "\n".join(filter(None, [perf_context, history_context, tech_context]))

    prompt = f"""You are a quantitative financial analyst.
{context_block + chr(10) if context_block else ""}The current closing price of {ticker} is ${current_price:.2f}.
Recent news:
{news_context}

Predict buy and sell price ranges for the NEXT trading session.
- buy_low / buy_high: price range to enter a long position (at or below current price, near support)
- sell_low / sell_high: price range to take profit (above current price, near resistance)

Respond ONLY with valid JSON and nothing else:
{{
  "buy_low": <number>,
  "buy_high": <number>,
  "sell_low": <number>,
  "sell_high": <number>,
  "rationale": "<one sentence>"
}}
All values must be positive numbers. buy_high must be less than sell_low."""

    for attempt in range(1, MAX_LLM_RETRIES + 1):
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model_name, "prompt": prompt, "stream": False,
                      "options": {"temperature": 0.1}},
                timeout=120,
            )
            response.raise_for_status()
            return parse_llm_range(response.json().get('response', '').strip(), current_price)
        except Exception as e:
            if attempt < MAX_LLM_RETRIES:
                print(f"    [!] Attempt {attempt}/{MAX_LLM_RETRIES} failed: {e}. Retrying in {LLM_RETRY_DELAY}s...")
                time.sleep(LLM_RETRY_DELAY)
            else:
                print(f"    [!] All {MAX_LLM_RETRIES} attempts failed. Using fallback range.")
                return _fallback_range(current_price)


def fetch_all_bars(tickers: list[str], days: int = 30) -> dict[str, list]:
    """Batch-fetch daily bars for all tickers. Returns {sym: [Bar, ...]} oldest->newest."""
    data_client = alpaca_client.get_data_client()
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days + 10)  # buffer for weekends/holidays
    result: dict[str, list] = {}
    for i in range(0, len(tickers), 200):
        batch = tickers[i:i + 200]
        try:
            req = StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame.Day,
                start=start.strftime('%Y-%m-%d'),
                end=end.strftime('%Y-%m-%d'),
                feed=DataFeed.IEX,
            )
            bars = data_client.get_stock_bars(req)
            for sym, sym_bars in bars.data.items():
                if sym_bars:
                    result[sym] = list(sym_bars)
        except Exception as e:
            print(f"  [!] Market data batch failed: {e}")
    return result


def git_checkpoint(message: str, paths: list[str]) -> bool:
    """Stage paths, commit if anything changed, and push with rebase+retry.

    Used between batches so partial results are pushed upstream and a crash
    mid-run never loses more than the last batch. Returns True on success
    (including the nothing-to-commit case), False if the push ultimately failed.
    """
    try:
        subprocess.run(['git', 'add', *paths], check=True)
        if subprocess.run(['git', 'diff', '--staged', '--quiet']).returncode == 0:
            return True  # nothing new to commit
        subprocess.run(['git', 'commit', '-m', message], check=True)
    except subprocess.CalledProcessError as e:
        print(f"  [!] git commit failed: {e}")
        return False

    for attempt in range(1, 6):
        subprocess.run(['git', 'rebase', '--abort'], capture_output=True)
        pulled = subprocess.run(['git', 'pull', '--rebase', '--autostash', 'origin', 'main'])
        if pulled.returncode == 0 and \
                subprocess.run(['git', 'push', 'origin', 'HEAD:main']).returncode == 0:
            print(f"  Pushed checkpoint on attempt {attempt}.")
            return True
        print(f"  [!] push attempt {attempt} failed; retrying in 5s...")
        time.sleep(5)
    print("  [!] Could not push checkpoint after 5 attempts.")
    return False


def main():
    parser = argparse.ArgumentParser(description='Run the OracleForge overnight inference loop.')
    parser.add_argument('--tickers', help='Comma-separated watchlist override.')
    parser.add_argument('--batch-size', type=int, default=25,
                        help='Tickers per checkpoint batch (saved/pushed incrementally). 0 = all at once.')
    parser.add_argument('--push', action='store_true',
                        help='git commit + push after each batch (CI use). Off by default.')
    args = parser.parse_args()

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting OracleForge Loop...")

    config_tickers = load_json(CONFIG_FILE, [])
    watchlist = parse_ticker_list(args.tickers)
    tickers = watchlist if watchlist else config_tickers
    journal = load_json(TRADE_JOURNAL_FILE, [])

    # Models come from config/models.json — never overwritten by automation
    models = load_json(MODELS_FILE, [])
    if not models:
        print("ERROR: No models found in config/models.json.")
        sys.exit(1)

    # Scores are per-model state — initialise any new model at 5.0
    scores = load_json(SCORES_FILE, {})
    for m in models:
        if m not in scores:
            print(f"  New model detected: {m} — initialising score to 5.0")
            scores[m] = 5.0
    # Restrict scores dict to active models only
    scores = {m: scores[m] for m in models}

    # Read trading config for score decay and stop threshold
    trading_cfg = load_json('config/trading.json', {})
    score_decay = float(trading_cfg.get('score_decay_per_day', 1.0))
    stop_loss_pct = float(trading_cfg.get('stop_loss_pct', 0.95))

    if not tickers:
        print("ERROR: No tickers found. Run update_tickers.py or pass --tickers NVDA,AAPL.")
        sys.exit(1)

    if watchlist:
        print(f"Watchlist mode: {len(tickers)} tickers ({', '.join(tickers)})")
    else:
        print(f"Processing {len(tickers)} tickers from {CONFIG_FILE}")
    print(f"Models: {', '.join(models)}")
    print(f"Trade journal: {len(journal)} closed trades on record.")
    print(f"Score decay per day: {score_decay}")

    today_date = datetime.now().strftime('%Y-%m-%d')
    today_log_path = os.path.join(HISTORY_DIR, f'predictions_{today_date}.json')
    report_path = os.path.join(REPORTS_DIR, f'signals_{today_date}.json')

    # Resume support: the score update is a once-per-day step, so it runs only
    # on the first invocation (when today's predictions file does not yet
    # exist). Tickers already present in that file are skipped this run, so a
    # crash/retry only re-processes what is left.
    first_run_today = not os.path.exists(today_log_path)
    existing_enriched = load_json(today_log_path, {})
    already_done = set(existing_enriched.keys())

    prior_log_path, prior_date = find_latest_predictions_path()
    prior_predictions = load_json(prior_log_path, {}) if prior_log_path else {}
    if prior_log_path:
        print(f"Evaluating predictions from {prior_date} ({prior_log_path})")
    else:
        print("No prior prediction file found; skipping evaluation this run.")

    # --- PHASE 1: MARKET DATA & TECHNICALS ---
    print("\n--- PHASE 1: Market Data ---")
    print(f"Batch-fetching 30-day bars for {len(tickers)} tickers via Alpaca...")
    all_bars = fetch_all_bars(tickers, days=30)
    print(f"  Received data for {len(all_bars)} tickers.")

    market_data: dict[str, float] = {}
    technicals_data: dict[str, dict] = {}
    for ticker in tickers:
        bar_list = all_bars.get(ticker)
        if not bar_list:
            continue
        market_data[ticker] = float(bar_list[-1].close)
        technicals_data[ticker] = compute_technicals(bar_list)

    if not market_data:
        print("ERROR: No market data retrieved for any ticker. Aborting inference.")
        sys.exit(1)

    # --- SCORE UPDATE (once per day, on the first invocation only) ---
    if first_run_today:
        score_deltas: dict = defaultdict(list)
        for ticker in tickers:
            bar_list = all_bars.get(ticker)
            if not bar_list:
                continue
            latest = bar_list[-1]
            prior_models = extract_model_predictions(prior_predictions.get(ticker, {}))
            for model_name, past_pred in prior_models.items():
                if model_name not in scores:
                    continue
                score_deltas[model_name].append(evaluate_range_prediction(
                    high_price=float(latest.high),
                    low_price=float(latest.low),
                    pred=past_pred,
                    stop_threshold=stop_loss_pct,
                ))
        if prior_date:
            journal_deltas = score_deltas_from_journal(journal, prior_date, scores)
            if journal_deltas:
                closed = sum(len(v) for v in journal_deltas.values())
                print(f"  Applying score updates from {closed} actual closed trade(s) on {prior_date}.")
                for model, deltas in journal_deltas.items():
                    score_deltas[model].extend(deltas)
        apply_score_deltas(scores, score_deltas, decay=score_decay)
        save_json(SCORES_FILE, scores)
        # Persist an (initially empty) predictions file as a "scoring done"
        # marker so a crash before the first batch does not re-apply deltas.
        save_json(today_log_path, existing_enriched)
        print("  Score update applied (first run today).")
    else:
        print(f"  Resuming: {len(already_done)} ticker(s) already processed today; skipping score update.")

    # --- PHASE 1b: EARNINGS SCREEN (pending tickers only) ---
    pending = [t for t in market_data if t not in already_done]
    if not pending:
        print("\nAll tickers already processed today. Nothing to do.")
        return
    print(f"\n--- PHASE 1b: Earnings screen ({len(pending)} pending) ---")
    earnings_skip: set[str] = set()
    for ticker in pending:
        if has_upcoming_earnings(ticker):
            earnings_skip.add(ticker)
    if earnings_skip:
        print(f"  Skipping {len(earnings_skip)} ticker(s) with earnings in {EARNINGS_LOOKAHEAD_DAYS} days: "
              f"{', '.join(sorted(earnings_skip))}")
    else:
        print("  No upcoming earnings found.")

    # --- PHASE 2 & 3: BATCHED INFERENCE + INCREMENTAL CHECKPOINTS ---
    print("\n--- PHASE 2: AI Inference (batched) ---")
    enriched_all = dict(existing_enriched)
    batch_size = args.batch_size if args.batch_size and args.batch_size > 0 else len(pending)
    fallback_count = 0
    total = len(market_data)

    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        active_batch = [t for t in batch if t not in earnings_skip]

        batch_raw: dict[str, dict] = {}
        # Earnings-skipped tickers still get a record (they classify as SKIP).
        for ticker in batch:
            if ticker in earnings_skip:
                batch_raw[ticker] = {m: {'skipped': True, 'reason': 'upcoming_earnings'} for m in scores}

        # Model-major order within the batch so Ollama loads each model once.
        for model_name in scores:
            if not active_batch:
                break
            print(f"  >> {model_name} (score {scores[model_name]:.3f}) on {len(active_batch)} ticker(s)")
            for ticker in active_batch:
                rp = call_local_llm(
                    ticker, market_data[ticker], model_name, journal,
                    technicals=technicals_data.get(ticker),
                )
                if rp.get('fallback'):
                    fallback_count += 1
                batch_raw.setdefault(ticker, {})[model_name] = rp

        # Enrich this batch, merge, and rebuild the report from everything so far.
        batch_closes = {t: market_data[t] for t in batch}
        batch_enriched = build_enriched_predictions(batch_raw, batch_closes, scores)
        enriched_all.update(batch_enriched)

        report = build_signals_report(enriched_all, today_date)
        save_json(today_log_path, enriched_all)
        save_signals_json(report_path, report)
        done = len(enriched_all)
        print(f"  Checkpoint saved: {done}/{total} tickers ({report['summary']['active']} ACTIVE so far).")

        if args.push:
            git_checkpoint(
                f"Forge batch {done}/{total} tickers ({today_date})",
                ['config/', SCORES_FILE, HISTORY_DIR, REPORTS_DIR],
            )

    if fallback_count:
        print(f"\n  [!] {fallback_count} fallback prediction(s) generated (excluded from consensus).")

    if not enriched_all:
        print("ERROR: No enriched predictions produced.")
        sys.exit(1)

    final_report = build_signals_report(enriched_all, today_date)
    print_signals_table(final_report)
    print(f"\nSaved predictions to {today_log_path}")
    print(f"Saved signals report to {report_path}")
    print(f"Generated {len(enriched_all)} ticker records ({final_report['summary']['active']} ACTIVE).")
    print("OracleForge Loop Complete.")


if __name__ == '__main__':
    main()
