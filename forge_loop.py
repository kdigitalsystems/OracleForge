# forge_loop.py
import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta

import requests
import yfinance as yf

from signals import (
    build_enriched_predictions,
    build_signals_report,
    extract_model_predictions,
    parse_ticker_list,
    print_signals_table,
    save_json as save_signals_json,
)

# --- Configuration ---
CONFIG_FILE = 'config/tickers.json'
SCORES_FILE = 'state/analyst_scores.json'
HISTORY_DIR = 'history/'
REPORTS_DIR = 'reports/'
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
    """Return the most recent predictions file (skips weekends/holidays)."""
    for days_back in range(1, MAX_PREDICTION_LOOKBACK_DAYS + 1):
        date_str = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        path = os.path.join(HISTORY_DIR, f'predictions_{date_str}.json')
        if os.path.exists(path):
            return path, date_str
    return None, None


def _headline_from_news_item(item):
    """Support legacy and current yfinance news payload shapes."""
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


def evaluate_prediction(open_price, high_price, low_price, predicted_price):
    """Return the score delta (+0.01 or -0.01) for one ticker evaluation."""
    stop_loss_limit = open_price * 0.98

    if low_price <= stop_loss_limit:
        return -0.01
    if high_price >= predicted_price:
        return 0.01
    return -0.01


def apply_score_deltas(scores, deltas_by_model):
    """Apply one aggregated score change per model for the trading day."""
    for model_name, deltas in deltas_by_model.items():
        if not deltas:
            continue
        current = scores.get(model_name, 5.0)
        # Average per-ticker deltas so 50 tickers do not swing scores by ±0.50/night.
        adjustment = sum(deltas) / len(deltas)
        scores[model_name] = round(min(10.0, max(0.0, current + adjustment)), 3)


def parse_llm_price(raw_output, fallback_price):
    """Extract the first plausible price from model output."""
    if not raw_output or not raw_output.strip():
        return round(fallback_price * 1.001, 2)

    match = re.search(r'\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+\.\d+', raw_output.replace(',', ''))
    if not match:
        return round(fallback_price * 1.001, 2)

    try:
        value = float(match.group().replace(',', ''))
        if value <= 0:
            return round(fallback_price * 1.001, 2)
        return round(value, 2)
    except ValueError:
        return round(fallback_price * 1.001, 2)


def call_local_llm(ticker, current_price, model_name):
    news_context = fetch_recent_news(ticker)

    prompt = f"""You are a quantitative financial analyst.
The current closing price of {ticker} is ${current_price:.2f}.
Here are the latest news headlines regarding this company:
{news_context}

Based on this data, predict the High of the Day (HOD) for the next trading session.
You must respond with ONLY the exact numerical price. Do not include dollar signs, words, or explanations. Example: 154.25"""

    try:
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1
            }
        }

        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()

        raw_output = response.json().get('response', '').strip()
        return parse_llm_price(raw_output, current_price)

    except Exception as e:
        print(f"    [!] AI generation failed. Error: {e}")
        return round(current_price * 1.001, 2)


def fetch_latest_session_bar(stock):
    """Return OHLC for the most recent completed trading session."""
    hist = stock.history(period='5d')
    if hist.empty:
        return None
    row = hist.iloc[-1]
    return {
        'open': float(row['Open']),
        'high': float(row['High']),
        'low': float(row['Low']),
        'close': float(row['Close']),
    }


def main():
    parser = argparse.ArgumentParser(description='Run the OracleForge daily inference loop.')
    parser.add_argument(
        '--tickers',
        help='Comma-separated watchlist (e.g. NVDA,AAPL,MSFT). Overrides config/tickers.json.',
    )
    args = parser.parse_args()

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting OracleForge Loop...")

    config_tickers = load_json(CONFIG_FILE, [])
    watchlist = parse_ticker_list(args.tickers)
    tickers = watchlist if watchlist else config_tickers
    scores = load_json(SCORES_FILE, {})

    if not tickers:
        print("ERROR: No tickers found. Run update_tickers.py or pass --tickers NVDA,AAPL.")
        sys.exit(1)

    if watchlist:
        print(f"Watchlist mode: {len(tickers)} tickers ({', '.join(tickers)})")
    else:
        print(f"Processing {len(tickers)} tickers from {CONFIG_FILE}")

    if not scores:
        print("ERROR: No models found in state/analyst_scores.json.")
        sys.exit(1)

    today_date = datetime.now().strftime('%Y-%m-%d')
    today_log_path = os.path.join(HISTORY_DIR, f'predictions_{today_date}.json')

    prior_log_path, prior_date = find_latest_predictions_path()
    prior_predictions = load_json(prior_log_path, {}) if prior_log_path else {}

    if prior_log_path:
        print(f"Evaluating predictions from {prior_date} ({prior_log_path})")
    else:
        print("No prior prediction file found; skipping evaluation this run.")

    today_predictions = {}
    market_data = {}
    score_deltas = defaultdict(list)

    # --- PHASE 1: FETCH ALL MARKET DATA & EVALUATE PRIOR PREDICTIONS ---
    print("\n--- PHASE 1: Market Data & Evaluation ---")
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            bar = fetch_latest_session_bar(stock)
            if bar is None:
                print(f"  Skipping {ticker}: no recent market data.")
                continue

            market_data[ticker] = bar['close']
            today_predictions[ticker] = {}

            prior_models = extract_model_predictions(prior_predictions.get(ticker, {}))
            for model_name, past_pred in prior_models.items():
                if model_name not in scores:
                    continue
                delta = evaluate_prediction(
                    open_price=bar['open'],
                    high_price=bar['high'],
                    low_price=bar['low'],
                    predicted_price=past_pred,
                )
                score_deltas[model_name].append(delta)

        except Exception as e:
            print(f"  Error fetching data for {ticker}: {e}")

    apply_score_deltas(scores, score_deltas)

    if not market_data:
        print("ERROR: No market data retrieved for any ticker. Aborting inference.")
        sys.exit(1)

    # --- PHASE 2: MODEL INFERENCE (HARDWARE OPTIMIZED) ---
    print("\n--- PHASE 2: AI Inference (Model by Model) ---")
    for model_name in scores.keys():
        print(f"\n>> Loading weights and processing with {model_name} <<")

        for ticker, current_price in market_data.items():
            print(f"  [{model_name}] Predicting {ticker} (Close: ${current_price:.2f})...")
            prediction = call_local_llm(ticker, current_price, model_name)
            today_predictions[ticker][model_name] = prediction

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
    print(f"Generated {len(enriched)} ticker records ({report['summary']['watch']} WATCH).")
    print("OracleForge Loop Complete. Ready for Git Commit.")


if __name__ == '__main__':
    main()
