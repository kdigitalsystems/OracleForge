# trader.py — daytime price monitor and order executor
"""
Polls live prices every N seconds during market hours.
Buys when price enters the buy range (up to $8 position, $2 at a time).
Sells the entire position when price hits the sell range.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, date

import pytz
from alpaca.data.requests import StockLatestBarRequest

import alpaca_client

REPORTS_DIR = 'reports/'
CONFIG_FILE = 'config/trading.json'

ET = pytz.timezone('America/New_York')

DEFAULT_TRADING_CONFIG = {
    'max_per_trade_usd': 2.0,
    'max_position_usd': 8.0,
    'poll_interval_sec': 60,
}


def load_json(filepath: str, default):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return default


def save_json(filepath: str, data) -> None:
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)


def load_trading_config() -> dict:
    config = load_json(CONFIG_FILE, {})
    return {**DEFAULT_TRADING_CONFIG, **config}


def is_market_open(cfg: dict) -> bool:
    now_et = datetime.now(ET)
    if now_et.weekday() >= 5:
        return False
    open_h, open_m = map(int, cfg.get('market_open', '09:30').split(':'))
    close_h, close_m = map(int, cfg.get('market_close', '16:00').split(':'))
    t = now_et.time()
    from datetime import time as dt_time
    return dt_time(open_h, open_m) <= t <= dt_time(close_h, close_m)


def get_prices(tickers: list[str]) -> dict[str, float]:
    """Batch-fetch latest bar price for all tickers via Alpaca."""
    try:
        data_client = alpaca_client.get_data_client()
        req = StockLatestBarRequest(symbol_or_symbols=tickers)
        bars = data_client.get_stock_latest_bar(req)
        return {sym: float(bar.close) for sym, bar in bars.items()}
    except Exception as e:
        print(f"  [!] Price fetch failed: {e}")
        return {}


def load_todays_signals() -> dict:
    """Load today's signals report. Returns empty dict if not yet generated."""
    today = date.today().strftime('%Y-%m-%d')
    path = os.path.join(REPORTS_DIR, f'signals_{today}.json')
    return load_json(path, {})


def load_trade_log(today: str) -> list[dict]:
    path = os.path.join(REPORTS_DIR, f'trades_{today}.json')
    return load_json(path, [])


def save_trade_log(today: str, log: list[dict]) -> None:
    path = os.path.join(REPORTS_DIR, f'trades_{today}.json')
    save_json(path, log)


def log_trade(trade_log: list, today: str, action: str, ticker: str, price: float, amount: float, reason: str) -> None:
    entry = {
        'timestamp': datetime.now(ET).isoformat(timespec='seconds'),
        'action': action,
        'ticker': ticker,
        'price': price,
        'amount_usd': amount,
        'reason': reason,
    }
    trade_log.append(entry)
    save_trade_log(today, trade_log)
    print(
        f"  [{entry['timestamp']}] {action} {ticker} @ ${price:.2f} "
        f"(${amount:.2f}) — {reason}"
    )


def run_trading_loop(dry_run: bool = False) -> None:
    cfg = load_trading_config()
    max_per_trade = float(cfg['max_per_trade_usd'])
    max_position = float(cfg['max_position_usd'])
    poll_interval = int(cfg['poll_interval_sec'])

    client = None if dry_run else alpaca_client.get_trading_client()
    today = date.today().strftime('%Y-%m-%d')
    trade_log = load_trade_log(today)

    print(f"[{datetime.now(ET).strftime('%Y-%m-%d %H:%M:%S %Z')}] OracleForge Trader started.")
    if dry_run:
        print("  -- DRY RUN MODE: no orders will be placed --")

    while True:
        if not is_market_open(cfg):
            now_et = datetime.now(ET)
            if now_et.weekday() >= 5 or now_et.hour >= 16:
                print(f"[{now_et.strftime('%H:%M %Z')}] Market closed. Trader shutting down.")
                break
            print(f"[{now_et.strftime('%H:%M %Z')}] Pre-market. Waiting for open...")
            time.sleep(60)
            continue

        signals = load_todays_signals()
        active_tickers = [row['ticker'] for row in signals.get('active', [])]

        if not active_tickers:
            print(f"[{datetime.now(ET).strftime('%H:%M')}] No ACTIVE setups today.")
            time.sleep(poll_interval)
            continue

        positions = {} if dry_run else alpaca_client.get_positions(client)

        active_rows = [r for r in signals.get('active', []) if r.get('buy_low') and r.get('buy_high') and r.get('sell_low')]
        if not active_rows:
            time.sleep(poll_interval)
            continue

        price_map = get_prices([r['ticker'] for r in active_rows])

        for row in active_rows:
            ticker = row['ticker']
            buy_low = row.get('buy_low')
            buy_high = row.get('buy_high')
            sell_low = row.get('sell_low')

            price = price_map.get(ticker)
            if price is None:
                print(f"  [!] Could not fetch price for {ticker}")
                continue

            position_value = positions.get(ticker, 0.0)

            # Sell logic — check first to free up capacity
            if price >= sell_low and position_value > 0:
                if dry_run:
                    log_trade(trade_log, today, 'SELL', ticker, price, position_value, 'Hit sell target (dry run)')
                else:
                    try:
                        alpaca_client.sell_all(client, ticker)
                        log_trade(trade_log, today, 'SELL', ticker, price, position_value, 'Hit sell target')
                        positions[ticker] = 0.0
                    except Exception as e:
                        print(f"  [!] Sell failed for {ticker}: {e}")
                continue

            # Buy logic — price must be in buy range with room for more position
            if buy_low <= price <= buy_high and position_value < max_position:
                order_size = min(max_per_trade, max_position - position_value)
                if order_size < 0.01:
                    continue

                if dry_run:
                    log_trade(trade_log, today, 'BUY', ticker, price, order_size, f'In buy range [{buy_low}-{buy_high}] (dry run)')
                else:
                    try:
                        alpaca_client.buy(client, ticker, order_size)
                        log_trade(trade_log, today, 'BUY', ticker, price, order_size, f'In buy range [{buy_low}-{buy_high}]')
                        positions[ticker] = position_value + order_size
                    except Exception as e:
                        print(f"  [!] Buy failed for {ticker}: {e}")

        time.sleep(poll_interval)


def main() -> None:
    parser = argparse.ArgumentParser(description='OracleForge daytime trader.')
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Log what would be traded without placing real orders.',
    )
    args = parser.parse_args()
    run_trading_loop(dry_run=args.dry_run)


if __name__ == '__main__':
    main()
