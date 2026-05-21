# trader.py — GTC limit order placement and end-of-day settlement
"""
Two short jobs replace the old polling loop:

  python trader.py --open   (9:30 AM ET)
    Reads today's ACTIVE signals and places DAY limit buy orders at buy_high.
    Also places GTC sell orders for any existing positions without one.

  python trader.py --close  (4:05 PM ET)
    Checks fills for open buy orders → records entry, places GTC sell.
    Checks fills for open sell orders → records P&L to trade journal.
    Cancels any unfilled DAY buy orders that expired.

State is persisted to state/open_orders.json so it survives between the two jobs.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date, datetime

import pytz

import alpaca_client

ORDER_SUBMIT_DELAY = 0.3   # seconds between order submissions (~200/min limit)

REPORTS_DIR = 'reports/'
HISTORY_DIR = 'history/'
CONFIG_FILE = 'config/trading.json'
OPEN_ORDERS_FILE = 'state/open_orders.json'
POSITIONS_META_FILE = 'state/open_positions_meta.json'
TRADE_JOURNAL_FILE = 'history/trade_journal.json'

ET = pytz.timezone('America/New_York')

DEFAULT_TRADING_CONFIG = {
    'max_per_trade_usd': 2.0,
    'max_position_usd': 8.0,
    'max_sell_order_days': 5,
}

FILLED_STATUSES = {'filled', 'partially_filled'}
DEAD_STATUSES = {'cancelled', 'expired', 'done_for_day', 'rejected', 'suspended'}


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
    return {**DEFAULT_TRADING_CONFIG, **load_json(CONFIG_FILE, {})}


def today_str() -> str:
    return date.today().strftime('%Y-%m-%d')


def now_et() -> str:
    return datetime.now(ET).isoformat(timespec='seconds')


def load_todays_signals() -> list[dict]:
    path = os.path.join(REPORTS_DIR, f'signals_{today_str()}.json')
    report = load_json(path, {})
    return [
        r for r in report.get('active', [])
        if r.get('buy_high') and r.get('sell_low')
    ]


def get_predicting_models(ticker: str, pred_date: str) -> list[str]:
    path = os.path.join(HISTORY_DIR, f'predictions_{pred_date}.json')
    predictions = load_json(path, {})
    models = predictions.get(ticker, {}).get('models', {})
    return [m for m, r in models.items() if isinstance(r, dict) and r.get('sell_low', 0) > 0]


def record_buy(meta: dict, ticker: str, price: float, usd_amount: float,
               pred_date: str, buy_high: float, sell_low: float) -> None:
    if ticker in meta:
        existing = meta[ticker]
        total = existing['usd_invested'] + usd_amount
        existing['entry_price'] = (
            existing['entry_price'] * existing['usd_invested'] + price * usd_amount
        ) / total
        existing['usd_invested'] = round(total, 4)
    else:
        meta[ticker] = {
            'entry_price': price,
            'usd_invested': round(usd_amount, 4),
            'entry_date': pred_date,
            'predicting_models': get_predicting_models(ticker, pred_date),
            'consensus_buy_high': buy_high,
            'consensus_sell_low': sell_low,
        }
    save_json(POSITIONS_META_FILE, meta)


def record_sell(meta: dict, journal: list, ticker: str,
                exit_price: float, usd_returned: float, close_date: str) -> dict | None:
    if ticker not in meta:
        return None
    entry = meta.pop(ticker)
    usd_invested = entry['usd_invested']
    usd_returned = round(usd_returned, 4)
    pnl_usd = round(usd_returned - usd_invested, 4)
    pnl_pct = round((pnl_usd / usd_invested) * 100, 2) if usd_invested else 0.0
    trade = {
        'close_date': close_date,
        'ticker': ticker,
        'entry_date': entry['entry_date'],
        'entry_price': entry['entry_price'],
        'exit_price': exit_price,
        'usd_invested': usd_invested,
        'usd_returned': usd_returned,
        'pnl_usd': pnl_usd,
        'pnl_pct': pnl_pct,
        'outcome': 'win' if pnl_usd > 0 else 'loss',
        'predicting_models': entry['predicting_models'],
        'consensus_buy_high': entry['consensus_buy_high'],
        'consensus_sell_low': entry['consensus_sell_low'],
    }
    journal.append(trade)
    save_json(TRADE_JOURNAL_FILE, journal)
    save_json(POSITIONS_META_FILE, meta)
    return trade


def log(msg: str) -> None:
    print(f"  [{now_et()}] {msg}")


# ---------------------------------------------------------------------------
# --open: place orders at market open
# ---------------------------------------------------------------------------

def run_open(dry_run: bool = False) -> None:
    cfg = load_trading_config()
    max_per_trade = float(cfg['max_per_trade_usd'])
    max_position = float(cfg['max_position_usd'])

    client = None if dry_run else alpaca_client.get_trading_client()
    today = today_str()

    active_rows = load_todays_signals()
    if not active_rows:
        print(f"No ACTIVE setups for {today}. Nothing to order.")
        return

    # Best upside first — ensures top setups are funded if buying power runs out
    active_rows.sort(key=lambda r: float(r.get('upside_pct', 0)), reverse=True)

    open_orders = load_json(OPEN_ORDERS_FILE, {})
    positions = {} if dry_run else alpaca_client.get_positions(client)
    positions_meta = load_json(POSITIONS_META_FILE, {})

    # Check available buying power upfront
    remaining_bp = float('inf')
    if not dry_run:
        try:
            account = client.get_account()
            remaining_bp = float(account.buying_power)
            print(f"  Available buying power: ${remaining_bp:.2f}")
        except Exception as e:
            print(f"  [!] Could not fetch buying power: {e} — will attempt orders anyway")

    print(f"[{now_et()}] OracleForge --open | {len(active_rows)} ACTIVE setups (sorted by upside)")
    if dry_run:
        print("  -- DRY RUN: no orders will be placed --")

    placed = 0

    # --- Place buy orders for ACTIVE tickers ---
    for row in active_rows:
        ticker = row['ticker']
        buy_high = float(row['buy_high'])
        sell_low = float(row['sell_low'])

        # Skip if we already placed a buy order today
        existing = open_orders.get(ticker, {})
        if existing.get('buy_order_id') and existing.get('date') == today:
            log(f"{ticker}: buy order already placed today, skipping")
            continue

        pos_val = positions.get(ticker, 0.0)
        if pos_val >= max_position:
            log(f"{ticker}: position ${pos_val:.2f} already at cap, skipping")
            continue

        order_size = min(max_per_trade, max_position - pos_val)
        qty = round(order_size / buy_high, 6)
        if qty < 0.001:
            log(f"{ticker}: order qty {qty} below minimum, skipping")
            continue

        if order_size > remaining_bp:
            log(f"Buying power exhausted (${remaining_bp:.2f} left, need ${order_size:.2f}). Stopping.")
            break

        if dry_run:
            log(f"DRY BUY {ticker}: DAY limit {qty} shares @ ${buy_high:.2f} (${order_size:.2f})")
            placed += 1
            continue

        try:
            order = alpaca_client.place_limit_buy(client, ticker, qty, buy_high, 'day')
            open_orders[ticker] = {
                'buy_order_id': str(order.id),
                'sell_order_id': None,
                'buy_limit': buy_high,
                'sell_limit': sell_low,
                'qty': qty,
                'date': today,
            }
            log(f"BUY {ticker}: DAY limit {qty} shares @ ${buy_high:.2f} (${order_size:.2f}) — order {order.id}")
            placed += 1
            remaining_bp -= order_size
            save_json(OPEN_ORDERS_FILE, open_orders)  # persist immediately — prevents duplicate orders on crash
            time.sleep(ORDER_SUBMIT_DELAY)
        except Exception as e:
            log(f"[!] Buy order failed for {ticker}: {e}")

    # --- Place sell orders for positions that don't have one yet ---
    for ticker, meta_entry in positions_meta.items():
        entry = open_orders.get(ticker, {})
        if entry.get('sell_order_id'):
            continue  # already have a sell order

        qty = 0.0 if dry_run else alpaca_client.get_position_qty(client, ticker)
        if qty <= 0:
            continue

        sell_limit = entry.get('sell_limit') or meta_entry.get('consensus_sell_low')
        if not sell_limit:
            continue

        if dry_run:
            log(f"DRY SELL {ticker}: GTC limit {qty} shares @ ${sell_limit:.2f}")
            continue

        try:
            order = alpaca_client.place_limit_sell(client, ticker, qty, sell_limit)
            if ticker not in open_orders:
                open_orders[ticker] = {}
            open_orders[ticker]['sell_order_id'] = str(order.id)
            log(f"SELL {ticker}: GTC limit {qty} shares @ ${sell_limit:.2f} — order {order.id}")
            time.sleep(ORDER_SUBMIT_DELAY)
        except Exception as e:
            log(f"[!] Sell order failed for {ticker}: {e}")

    if not dry_run:
        save_json(OPEN_ORDERS_FILE, open_orders)

    print(f"\nDone. {placed} buy order(s) placed.")


# ---------------------------------------------------------------------------
# --close: settle fills and update journal
# ---------------------------------------------------------------------------

def run_close(dry_run: bool = False) -> None:
    cfg = load_trading_config()
    max_sell_order_days = int(cfg.get('max_sell_order_days', 5))

    client = None if dry_run else alpaca_client.get_trading_client()
    today = today_str()

    open_orders = load_json(OPEN_ORDERS_FILE, {})
    positions_meta = load_json(POSITIONS_META_FILE, {})
    journal = load_json(TRADE_JOURNAL_FILE, [])

    if not open_orders:
        print(f"[{now_et()}] No open orders on record. Nothing to settle.")
        return

    print(f"[{now_et()}] OracleForge --close | settling {len(open_orders)} tracked ticker(s)")
    if dry_run:
        print("  -- DRY RUN: no state will be written --")

    # Fetch all recent orders in one API call instead of one per ticker
    orders_by_id: dict[str, object] = {}
    if not dry_run:
        try:
            all_orders = alpaca_client.get_all_recent_orders(client)
            orders_by_id = {str(o.id): o for o in all_orders}
            print(f"  Fetched {len(orders_by_id)} recent order(s) from Alpaca.")
        except Exception as e:
            print(f"  [!] Could not fetch orders from Alpaca: {e}")

    to_delete = []

    for ticker, entry in open_orders.items():
        buy_oid = entry.get('buy_order_id')
        sell_oid = entry.get('sell_order_id')

        # --- Check buy order ---
        if buy_oid and not sell_oid:
            if dry_run:
                log(f"DRY CHECK buy order {buy_oid} for {ticker}")
            else:
                order = orders_by_id.get(buy_oid)
                if order is None:
                    log(f"[!] Buy order {buy_oid} for {ticker} not found in recent orders")
                else:
                    status = str(order.status).lower().replace('orderstatus.', '')

                    if status in FILLED_STATUSES:
                        fill_price = float(order.filled_avg_price or entry['buy_limit'])
                        fill_qty = float(order.filled_qty or entry['qty'])
                        usd_filled = round(fill_price * fill_qty, 4)

                        record_buy(
                            positions_meta, ticker, fill_price, usd_filled,
                            entry.get('date', today),
                            entry['buy_limit'], entry['sell_limit'],
                        )
                        log(f"BUY FILLED {ticker}: {fill_qty} shares @ ${fill_price:.2f} (${usd_filled:.2f})")

                        try:
                            sell_order = alpaca_client.place_limit_sell(
                                client, ticker, fill_qty, entry['sell_limit']
                            )
                            entry['sell_order_id'] = str(sell_order.id)
                            log(f"SELL {ticker}: GTC limit {fill_qty} shares @ ${entry['sell_limit']:.2f} — order {sell_order.id}")
                            time.sleep(ORDER_SUBMIT_DELAY)
                        except Exception as e:
                            log(f"[!] Could not place sell for {ticker}: {e}")

                    elif status in DEAD_STATUSES:
                        log(f"BUY EXPIRED {ticker}: order {buy_oid} status={status}, removing")
                        to_delete.append(ticker)

                    else:
                        log(f"{ticker}: buy order {buy_oid} still open (status={status})")

        # --- Check sell order ---
        if sell_oid:
            if dry_run:
                log(f"DRY CHECK sell order {sell_oid} for {ticker}")
            else:
                order = orders_by_id.get(sell_oid)
                if order is None:
                    log(f"[!] Sell order {sell_oid} for {ticker} not found — may be older GTC, skipping")
                else:
                    status = str(order.status).lower().replace('orderstatus.', '')

                    if status in FILLED_STATUSES:
                        fill_price = float(order.filled_avg_price or entry['sell_limit'])
                        fill_qty = float(order.filled_qty or entry.get('qty', 0))
                        usd_returned = round(fill_price * fill_qty, 4)

                        trade = record_sell(
                            positions_meta, journal, ticker, fill_price, usd_returned, today
                        )
                        if trade:
                            log(
                                f"SELL FILLED {ticker}: {fill_qty} shares @ ${fill_price:.2f} "
                                f"— P&L ${trade['pnl_usd']:+.4f} ({trade['pnl_pct']:+.2f}%)"
                            )
                        to_delete.append(ticker)

                    else:
                        order_date = entry.get('date', today)
                        try:
                            days_old = (date.fromisoformat(today) - date.fromisoformat(order_date)).days
                        except Exception:
                            days_old = 0
                        if days_old >= max_sell_order_days:
                            alpaca_client.cancel_order(client, sell_oid)
                            entry['sell_order_id'] = None
                            log(f"CANCELLED stale sell order for {ticker} "
                                f"({days_old}d old, limit={max_sell_order_days}d) — will re-price tomorrow")
                        else:
                            log(f"{ticker}: sell GTC carries over (status={status}, age={days_old}d)")

    for ticker in set(to_delete):
        open_orders.pop(ticker, None)

    if not dry_run:
        save_json(OPEN_ORDERS_FILE, open_orders)

    print(f"\nDone. {len(set(to_delete))} position(s) closed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description='OracleForge GTC order manager.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--open', action='store_true', help='Place limit buy orders (run at market open).')
    group.add_argument('--close', action='store_true', help='Settle fills and update journal (run after close).')
    parser.add_argument('--dry-run', action='store_true', help='Log actions without placing or recording anything.')
    args = parser.parse_args()

    if args.open:
        run_open(dry_run=args.dry_run)
    else:
        run_close(dry_run=args.dry_run)


if __name__ == '__main__':
    main()
