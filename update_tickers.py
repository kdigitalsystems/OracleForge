# update_tickers.py
"""
Build the overnight watchlist from Alpaca's full US equity universe.

Filtering pipeline:
  1. All active, tradable, fractionable US equities from Alpaca (~2-4k symbols)
  2. Drop leveraged/inverse/VIX-futures funds by name (structural decay, not
     suited to a multi-day swing hold with symmetric risk/reward assumptions)
  3. Pre-filter: latest price >= min_price AND avg daily volume >= min_avg_daily_volume
  4. Volatility filter: 20-day daily return std dev <= max_daily_volatility_pct
  5. Sort by avg daily volume (most liquid first), take top max_tickers
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from datetime import datetime, timedelta, timezone

from alpaca.data.enums import DataFeed
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.enums import AssetClass, AssetStatus
from alpaca.trading.requests import GetAssetsRequest

import alpaca_client

UNIVERSE_CONFIG_FILE = 'config/universe.json'
TICKERS_FILE = 'config/tickers.json'

DEFAULT_CONFIG = {
    'min_price': 10.0,
    'min_avg_daily_volume': 500_000,
    'max_daily_volatility_pct': 4.0,
    'max_tickers': 200,
    'volatility_lookback_days': 20,
}

BATCH_SIZE = 200
BATCH_SLEEP = 0.3

# Leveraged/inverse funds and VIX-futures products decay structurally over
# time (daily rebalancing drag, contango) and don't behave like normal
# equities on a multi-day swing hold -- they're excluded by fund name rather
# than ticker so new products launched later are still caught.
LEVERAGED_NAME_PATTERN = re.compile(
    r'\bvix\b|\bultra(short|pro)?\b|\bleveraged\b|\binverse\b|\bdaily target\b'
    r'|\b\d(\.\d)?x\b',
    re.IGNORECASE,
)
LEVERAGED_ISSUER_PATTERN = re.compile(
    r'\bproshares\b|\bdirexion\b|\bgraniteshares\b|\bvolatility shares\b'
    r'|\bt-rex\b|\btradr\b|\brex shares\b|\bmicrosectors\b|\bdefiance daily\b',
    re.IGNORECASE,
)


def is_leveraged_or_inverse(name: str) -> bool:
    """Heuristic: fund name mentions leverage/inverse/VIX, or a known
    leveraged-fund issuer combined with a directional keyword (e.g.
    "ProShares Short S&P500")."""
    if not name:
        return False
    if LEVERAGED_NAME_PATTERN.search(name):
        return True
    if LEVERAGED_ISSUER_PATTERN.search(name) and re.search(
        r'\bshort\b|\bbull\b|\bbear\b|\blong\b', name, re.IGNORECASE
    ):
        return True
    return False


def load_universe_config() -> dict:
    if os.path.exists(UNIVERSE_CONFIG_FILE):
        with open(UNIVERSE_CONFIG_FILE) as f:
            saved = json.load(f)
        return {**DEFAULT_CONFIG, **saved}
    return DEFAULT_CONFIG.copy()


def get_tradable_symbols(trading_client) -> list[str]:
    print('Fetching Alpaca asset universe...')
    assets = trading_client.get_all_assets(
        GetAssetsRequest(
            asset_class=AssetClass.US_EQUITY,
            status=AssetStatus.ACTIVE,
        )
    )
    tradable = [a for a in assets if a.tradable and a.fractionable]
    symbols = [a.symbol for a in tradable if not is_leveraged_or_inverse(getattr(a, 'name', ''))]
    excluded = len(tradable) - len(symbols)
    print(f'  {len(tradable):,} active tradable fractionable US equity assets found '
          f'({excluded:,} excluded as leveraged/inverse/VIX-futures)')
    return symbols


def fetch_daily_bars(data_client, symbols: list[str], days: int) -> dict[str, list]:
    """Batch-fetch daily OHLCV bars for all symbols. Returns {symbol: [Bar, ...]}."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days + 10)  # buffer for weekends/holidays

    result: dict[str, list] = {}
    total_batches = (len(symbols) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f'  Batch {batch_num}/{total_batches} ({len(batch)} symbols)...', end='\r')
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
                result[sym] = list(sym_bars)
        except Exception as e:
            print(f'\n  [!] Batch {batch_num} failed: {e}')
        time.sleep(BATCH_SLEEP)

    print()  # newline after the \r progress
    return result


def compute_stats(bars: list) -> dict | None:
    """Return price, avg_volume, daily_vol_pct or None if insufficient data."""
    if len(bars) < 5:
        return None
    closes = [b.close for b in bars]
    volumes = [b.volume for b in bars]
    returns = [(closes[j] - closes[j - 1]) / closes[j - 1] for j in range(1, len(closes))]
    if len(returns) < 2:
        return None
    return {
        'price': closes[-1],
        'avg_volume': sum(volumes) / len(volumes),
        'daily_vol_pct': statistics.stdev(returns) * 100,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Update OracleForge ticker universe from Alpaca.')
    parser.add_argument('--limit', type=int, help='Override max_tickers')
    parser.add_argument('--min-price', type=float, help='Override min price (default $10)')
    parser.add_argument('--max-vol', type=float, help='Override max daily volatility %% (default 4.0)')
    args = parser.parse_args()

    cfg = load_universe_config()
    if args.limit:
        cfg['max_tickers'] = args.limit
    if args.min_price is not None:
        cfg['min_price'] = args.min_price
    if args.max_vol is not None:
        cfg['max_daily_volatility_pct'] = args.max_vol

    min_price = float(cfg['min_price'])
    min_volume = float(cfg['min_avg_daily_volume'])
    max_vol_pct = float(cfg['max_daily_volatility_pct'])
    max_tickers = int(cfg['max_tickers'])
    lookback = int(cfg['volatility_lookback_days'])

    print(f'Universe config: price >= ${min_price}, volume >= {min_volume:,.0f}, '
          f'volatility <= {max_vol_pct}%, limit = {max_tickers}')

    trading_client = alpaca_client.get_trading_client()
    data_client = alpaca_client.get_data_client()

    # Step 1: Full universe from Alpaca
    all_symbols = get_tradable_symbols(trading_client)

    # Step 2: Pre-filter by price and volume (10-day bars)
    print(f'\nFetching 10-day daily bars for {len(all_symbols):,} symbols...')
    bars_10d = fetch_daily_bars(data_client, all_symbols, days=10)

    candidates = []
    for sym, bars in bars_10d.items():
        stats = compute_stats(bars)
        if stats and stats['price'] >= min_price and stats['avg_volume'] >= min_volume:
            candidates.append(sym)

    print(f'  {len(candidates)} passed: price >= ${min_price} and avg volume >= {min_volume:,.0f}')

    if not candidates:
        print('ERROR: No candidates after pre-filter. Check thresholds.')
        sys.exit(1)

    # Step 3: Volatility filter (longer lookback, candidates only)
    print(f'\nFetching {lookback}-day daily bars for {len(candidates)} candidates...')
    bars_long = fetch_daily_bars(data_client, candidates, days=lookback)

    scored = []
    for sym in candidates:
        bars = bars_long.get(sym) or bars_10d.get(sym)
        stats = compute_stats(bars or [])
        if stats is None:
            continue
        if stats['daily_vol_pct'] > max_vol_pct:
            continue
        scored.append({
            'ticker': sym,
            'price': stats['price'],
            'avg_volume': stats['avg_volume'],
            'daily_vol_pct': round(stats['daily_vol_pct'], 2),
        })

    print(f'  {len(scored)} passed volatility <= {max_vol_pct}% daily std dev')

    if not scored:
        print('ERROR: No tickers passed all filters.')
        sys.exit(1)

    # Step 4: Sort by avg volume (most liquid first), take top N
    scored.sort(key=lambda x: x['avg_volume'], reverse=True)
    final = [s['ticker'] for s in scored[:max_tickers]]

    os.makedirs('config', exist_ok=True)
    with open(TICKERS_FILE, 'w') as f:
        json.dump(final, f, indent=4)

    print(f'\nSaved {len(final)} tickers to {TICKERS_FILE}')
    print(f'Top 10 by volume: {", ".join(final[:10])}')
    print(f'Price range: ${min(s["price"] for s in scored[:max_tickers]):.2f} – '
          f'${max(s["price"] for s in scored[:max_tickers]):.2f}')
    print(f'Volatility range: {min(s["daily_vol_pct"] for s in scored[:max_tickers]):.2f}% – '
          f'{max(s["daily_vol_pct"] for s in scored[:max_tickers]):.2f}% daily std dev')


if __name__ == '__main__':
    main()
