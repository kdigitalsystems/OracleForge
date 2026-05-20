# forge_loop.py — overnight inference engine
import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta

import requests
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
SCORES_FILE = 'state/analyst_scores.json'
HISTORY_DIR = 'history/'
REPORTS_DIR = 'reports/'
TRADE_JOURNAL_FILE = 'history/trade_journal.json'
MAX_PREDICTION_LOOKBACK_DAYS = 10

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
        import yfinance as yf
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


def _fallback_range(price: float) -> dict:
    return {
        'buy_low': round(price * 0.985, 2),
        'buy_high': round(price * 0.995, 2),
        'sell_low': round(price * 1.015, 2),
        'sell_high': round(price * 1.025, 2),
        'rationale': 'Fallback range (model output unparseable)',
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


def evaluate_range_prediction(open_price, high_price, low_price, pred) -> float:
    """Score a prior range prediction against realized OHLC: +0.01, -0.01, or 0.0."""
    if not isinstance(pred, dict):
        return 0.0
    buy_high = float(pred.get('buy_high') or 0)
    sell_low = float(pred.get('sell_low') or 0)
    if buy_high <= 0 or sell_low <= 0:
        return 0.0
    if low_price > buy_high:
        return 0.0
    stop_limit = buy_high * 0.98
    if low_price <= stop_limit:
        return -0.01
    if high_price >= sell_low:
        return 0.01
    return -0.01


def score_deltas_from_journal(journal: list, prior_date: str, scores: dict) -> dict:
    """
    Build score deltas from actual closed trades on prior_date.
    Models that predicted a winning trade get +0.02; losers get -0.02.
    Weighted higher than theoretical OHLC evaluation (±0.01) to reflect real outcomes.
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


def apply_score_deltas(scores: dict, deltas_by_model: dict) -> None:
    for model_name, deltas in deltas_by_model.items():
        non_zero = [d for d in deltas if d != 0.0]
        if not non_zero:
            continue
        current = scores.get(model_name, 5.0)
        adjustment = sum(non_zero) / len(non_zero)
        scores[model_name] = round(min(10.0, max(0.0, current + adjustment)), 3)


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
            f"  {t['close_date']}: bought ${t['entry_price']:.2f} → "
            f"sold ${t['exit_price']:.2f}  {t['pnl_pct']:+.2f}% ({t['outcome']})"
        )
    return "\n".join(lines)


def call_local_llm(ticker: str, current_price: float, model_name: str,
                   journal: list | None = None) -> dict:
    journal = journal or []
    news_context = fetch_recent_news(ticker)
    perf_context = model_performance_context(model_name, journal)
    history_context = ticker_trade_history(ticker, journal)

    context_block = "\n".join(filter(None, [perf_context, history_context]))

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
        print(f"    [!] AI generation failed. Error: {e}")
        return _fallback_range(current_price)


def fetch_all_bars(tickers: list[str]) -> dict[str, dict]:
    """Batch-fetch the latest daily session bar for all tickers via Alpaca."""
    data_client = alpaca_client.get_data_client()
    end = datetime.now()
    start = end - timedelta(days=10)
    result: dict[str, dict] = {}
    for i in range(0, len(tickers), 200):
        batch = tickers[i:i + 200]
        try:
            req = StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame.Day,
                start=start.strftime('%Y-%m-%d'),
                end=end.strftime('%Y-%m-%d'),
            )
            bars = data_client.get_stock_bars(req)
            for sym, sym_bars in bars.data.items():
                if sym_bars:
                    b = sym_bars[-1]
                    result[sym] = {
                        'open': float(b.open), 'high': float(b.high),
                        'low': float(b.low), 'close': float(b.close),
                    }
        except Exception as e:
            print(f"  [!] Market data batch failed: {e}")
    return result


def main():
    parser = argparse.ArgumentParser(description='Run the OracleForge overnight inference loop.')
    parser.add_argument('--tickers', help='Comma-separated watchlist override.')
    args = parser.parse_args()

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting OracleForge Loop...")

    config_tickers = load_json(CONFIG_FILE, [])
    watchlist = parse_ticker_list(args.tickers)
    tickers = watchlist if watchlist else config_tickers
    scores = load_json(SCORES_FILE, {})
    journal = load_json(TRADE_JOURNAL_FILE, [])

    if not tickers:
        print("ERROR: No tickers found. Run update_tickers.py or pass --tickers NVDA,AAPL.")
        sys.exit(1)
    if not scores:
        print("ERROR: No models found in state/analyst_scores.json.")
        sys.exit(1)

    if watchlist:
        print(f"Watchlist mode: {len(tickers)} tickers ({', '.join(tickers)})")
    else:
        print(f"Processing {len(tickers)} tickers from {CONFIG_FILE}")
    print(f"Trade journal: {len(journal)} closed trades on record.")

    today_date = datetime.now().strftime('%Y-%m-%d')
    today_log_path = os.path.join(HISTORY_DIR, f'predictions_{today_date}.json')

    prior_log_path, prior_date = find_latest_predictions_path()
    prior_predictions = load_json(prior_log_path, {}) if prior_log_path else {}

    if prior_log_path:
        print(f"Evaluating predictions from {prior_date} ({prior_log_path})")
    else:
        print("No prior prediction file found; skipping evaluation this run.")

    today_predictions: dict[str, dict] = {}
    market_data: dict[str, float] = {}
    score_deltas: dict = defaultdict(list)

    # --- PHASE 1: FETCH MARKET DATA & EVALUATE PRIOR PREDICTIONS ---
    print("\n--- PHASE 1: Market Data & Evaluation ---")
    print(f"Batch-fetching latest bars for {len(tickers)} tickers via Alpaca...")
    all_bars = fetch_all_bars(tickers)
    print(f"  Received data for {len(all_bars)} tickers.")

    for ticker in tickers:
        bar = all_bars.get(ticker)
        if bar is None:
            print(f"  Skipping {ticker}: no recent market data.")
            continue

        market_data[ticker] = bar['close']
        today_predictions[ticker] = {}

        # Theoretical OHLC-based score update (range quality check)
        prior_models = extract_model_predictions(prior_predictions.get(ticker, {}))
        for model_name, past_pred in prior_models.items():
            if model_name not in scores:
                continue
            delta = evaluate_range_prediction(
                open_price=bar['open'], high_price=bar['high'],
                low_price=bar['low'], pred=past_pred,
            )
            score_deltas[model_name].append(delta)

    # Actual P&L-based score update from trade journal (overrides OHLC for traded tickers)
    if prior_date:
        journal_deltas = score_deltas_from_journal(journal, prior_date, scores)
        if journal_deltas:
            closed = sum(len(v) for v in journal_deltas.values())
            print(f"  Applying score updates from {closed} actual closed trade(s) on {prior_date}.")
            # Journal outcomes are weighted higher — merge with higher priority
            for model, deltas in journal_deltas.items():
                score_deltas[model].extend(deltas)

    apply_score_deltas(scores, score_deltas)

    if not market_data:
        print("ERROR: No market data retrieved for any ticker. Aborting inference.")
        sys.exit(1)

    # --- PHASE 2: MODEL INFERENCE ---
    print("\n--- PHASE 2: AI Inference (Model by Model) ---")
    for model_name in scores.keys():
        print(f"\n>> Loading {model_name} (score: {scores[model_name]:.3f}) <<")
        for ticker, current_price in market_data.items():
            print(f"  [{model_name}] Predicting {ticker} (Close: ${current_price:.2f})...")
            range_pred = call_local_llm(ticker, current_price, model_name, journal)
            today_predictions[ticker][model_name] = range_pred

    # --- PHASE 3: SIGNALS & SAVE STATE ---
    print("\n--- PHASE 3: Signals & Persistence ---")
    enriched = build_enriched_predictions(today_predictions, market_data, scores)
    report = build_signals_report(enriched, today_date)
    report_path = os.path.join(REPORTS_DIR, f'signals_{today_date}.json')

    if not enriched:
        print("ERROR: No enriched predictions produced.")
        sys.exit(1)

    save_json(SCORES_FILE, scores)
    save_json(today_log_path, enriched)
    save_signals_json(report_path, report)
    print_signals_table(report)
    print(f"\nSaved predictions to {today_log_path}")
    print(f"Saved signals report to {report_path}")
    print(f"Generated {len(enriched)} ticker records ({report['summary']['active']} ACTIVE).")
    print("OracleForge Loop Complete. Ready for Git Commit.")


if __name__ == '__main__':
    main()
