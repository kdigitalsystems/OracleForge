# update_tickers.py
import argparse
import io
import json
import os
import time

import pandas as pd
import requests
import yfinance as yf

CONFIG_FILE = 'config/tickers.json'


def fetch_top_tickers(limit):
    print("Scraping current S&P 500 constituents...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    html_data = io.StringIO(response.text)
    tables = pd.read_html(html_data)
    sp500_df = tables[0]

    raw_tickers = sp500_df['Symbol'].str.replace('.', '-', regex=False).tolist()

    print(
        "Fetching market capitalization data... "
        "(This will take about a minute to avoid rate limits)"
    )
    ticker_data = []

    for ticker in raw_tickers:
        try:
            stock = yf.Ticker(ticker)
            try:
                market_cap = stock.fast_info['marketCap']
            except (KeyError, TypeError, AttributeError):
                market_cap = stock.info.get('marketCap', 0)

            if market_cap and market_cap > 0:
                ticker_data.append({'ticker': ticker, 'market_cap': market_cap})

        except Exception:
            print(f"  Skipping {ticker} due to data fetch error.")

        time.sleep(0.1)

    if not ticker_data:
        existing = []
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                existing = json.load(f)
        if existing:
            print(
                "ERROR: Could not fetch market cap data. "
                f"Leaving existing {len(existing)} tickers in {CONFIG_FILE}."
            )
        else:
            print("ERROR: Could not fetch market cap data and no existing ticker list to keep.")
        return

    sorted_tickers = sorted(ticker_data, key=lambda x: x['market_cap'], reverse=True)
    top_tickers = [item['ticker'] for item in sorted_tickers[:limit]]

    os.makedirs('config', exist_ok=True)

    with open(CONFIG_FILE, 'w') as f:
        json.dump(top_tickers, f, indent=4)

    print(f"\nSuccessfully saved top {len(top_tickers)} tickers to {CONFIG_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update the ticker list dynamically.")
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Number of top tickers to retrieve (default: 50).",
    )
    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be at least 1")

    fetch_top_tickers(args.limit)
