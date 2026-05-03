# update_tickers.py
import pandas as pd
import yfinance as yf
import requests
import json
import argparse
import os
import io
import time

def fetch_top_tickers(limit):
    print("Scraping current S&P 500 constituents...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    html_data = io.StringIO(response.text)
    tables = pd.read_html(html_data)
    sp500_df = tables[0]
    
    raw_tickers = sp500_df['Symbol'].str.replace('.', '-', regex=False).tolist()
    
    print("Fetching market capitalization data... (This will take about 60 seconds to avoid rate limits)")
    ticker_data = []
    
    for ticker in raw_tickers:
        try:
            stock = yf.Ticker(ticker)
            # Try fast_info first (newer yfinance versions handle this differently)
            try:
                market_cap = stock.fast_info['marketCap']
            except (KeyError, TypeError, AttributeError):
                # Fallback to standard info fetch
                market_cap = stock.info.get('marketCap', 0)

            if market_cap > 0:
                ticker_data.append({'ticker': ticker, 'market_cap': market_cap})
                
        except Exception as e:
            print(f"  Skipping {ticker} due to data fetch error.")
            
        # The crucial fix: Be polite to Yahoo's servers
        time.sleep(0.1) 

    sorted_tickers = sorted(ticker_data, key=lambda x: x['market_cap'], reverse=True)
    top_tickers = [item['ticker'] for item in sorted_tickers[:limit]]
    
    os.makedirs('config', exist_ok=True)
    
    with open('config/tickers.json', 'w') as f:
        json.dump(top_tickers, f, indent=4)
        
    print(f"\nSuccessfully saved top {limit} tickers to config/tickers.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update the ticker list dynamically.")
    parser.add_argument("--limit", type=int, default=50, help="Number of top tickers to retrieve.")
    args = parser.parse_args()
    fetch_top_tickers(args.limit)
