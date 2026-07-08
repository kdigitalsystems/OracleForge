# trader.py — DAY limit order placement and end-of-day settlement
"""
Two short jobs replace the old polling loop:

  python3 trader.py --open   (9:30 AM ET)
    Reads today's ACTIVE signals and places DAY limit buy orders at buy_high.
    Also places DAY limit sell and stop-loss orders for positions without them.

  python3 trader.py --close  (4:05 PM ET)
    Checks fills for open buy orders -> records entry, places sell + stop orders.
    Checks fills for sell/stop orders -> records P&L, cancels companion order.
    Clears expired DAY orders so --open re-places them next morning.

State is persisted to state/open_orders.json so it survives between the two jobs.

Note: Alpaca does not support GTC for fractional share orders; all orders use DAY
time-in-force and are re-placed each morning until the position is exited.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timedelta

import pytz

import alpaca_client

ORDER_SUBMIT_DELAY = 0.3   # seconds between order submissions (~200/min limit)
# The nightly forge runs the evening before the trading day and names its
# output files with its own date, so the morning --open job must read the most
# recent signals file rather than strictly "today". Look back enough days to
# bridge weekends and holidays (Fri night -> Tue morning after a Mon holiday).
SIGNALS_MAX_LOOKBACK_DAYS = 4

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
    'stop_loss_pct': 0.95,
    'max_hold_days': 15,
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
    return datetime.now(ET).strftime('%Y-%m-%d')  # always ET to match signal files


def now_et() -> str:
    return datetime.now(ET).isoformat(timespec='seconds')


def find_latest_signals() -> tuple[str | None, str | None]:
    """Return (path, date_str) of the most recent signals file within the window.

    The signals file is dated by the nightly forge run (evening before the
    trading day), so the morning job reads the newest available file rather
    than one keyed to the current calendar date.
    """
    base = datetime.now(ET)
    for days_back in range(0, SIGNALS_MAX_LOOKBACK_DAYS + 1):
        date_str = (base - timedelta(days=days_back)).strftime('%Y-%m-%d')
        path = os.path.join(REPORTS_DIR, f'signals_{date_str}.json')
        if os.path.exists(path):
            return path, date_str
    return None, None


def load_todays_signals() -> tuple[list[dict], str | None]:
    """Load ACTIVE rows from the most recent signals file. Returns (rows, date)."""
    path, date_str = find_latest_signals()
    if not path:
        return [], None
    report = load_json(path, {})
    rows = [
        r for r in report.get('active', [])
        if r.get('buy_high') and r.get('sell_low')
    ]
    return rows, date_str


def get_predicting_models(ticker: str, pred_date: str) -> list[str]:
    path = os.path.join(HISTORY_DIR, f'predictions_{pred_date}.json')
    predictions = load_json(path, {})
    models = predictions.get(ticker, {}).get('models', {})
    # Exclude fallback (synthetic) predictions: they're dropped from the
    # consensus, so they didn't drive the trade and must not be credited or
    # penalised when the position closes (keeps the score feedback honest).
    return [
        m for m, r in models.items()
        if isinstance(r, dict) and r.get('sell_low', 0) > 0 and not r.get('fallback')
    ]


def record_buy(meta: dict, ticker: str, price: float, usd_amount: float,
               pred_date: str, buy_high: float, sell_low: float) -> None:
    if ticker in meta:
        existing = meta[ticker]
        total = existing['usd_invested'] + usd_amount
        # Share-weighted average cost basis: total dollars / total shares.
        # (A USD-weighted average of the two *prices* is not the cost basis.)
        old_entry = existing.get('entry_price') or 0.0
        old_shares = existing['usd_invested'] / old_entry if old_entry else 0.0
        new_shares = usd_amount / price if price else 0.0
        total_shares = old_shares + new_shares
        existing['entry_price'] = round(total / total_shares, 4) if total_shares else price
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
                exit_price: float, usd_returned: float, close_date: str,
                close_fraction: float = 1.0) -> dict | None:
    if ticker not in meta:
        return None
    entry = meta[ticker]
    close_fraction = min(1.0, max(0.0, float(close_fraction)))
    if close_fraction <= 0:
        return None
    usd_invested = round(entry['usd_invested'] * close_fraction, 4)
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
    if close_fraction >= 0.999999:
        meta.pop(ticker, None)
    else:
        entry['usd_invested'] = round(entry['usd_invested'] - usd_invested, 4)
    save_json(TRADE_JOURNAL_FILE, journal)
    save_json(POSITIONS_META_FILE, meta)
    return trade


def _filled_qty(order, fallback: float = 0.0) -> float:
    try:
        return float(order.filled_qty or fallback or 0.0)
    except (TypeError, ValueError):
        return float(fallback or 0.0)


def _close_fraction(fill_qty: float, total_qty: float) -> float:
    if total_qty <= 0:
        return 1.0
    return min(1.0, max(0.0, fill_qty / total_qty))


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

    active_rows, signals_date = load_todays_signals()
    if not active_rows:
        print(
            f"No ACTIVE setups found (searched last {SIGNALS_MAX_LOOKBACK_DAYS} days). "
            "Will still protect existing positions."
        )
    else:
        print(f"  Using signals from {signals_date} ({len(active_rows)} ACTIVE).")

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

        # Skip if we already hold a position from a prior signal (no unintended pyramiding)
        if ticker in positions_meta:
            log(f"{ticker}: position already held (${positions_meta[ticker]['usd_invested']:.2f} invested), skipping buy")
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
                'stop_order_id': None,
                'buy_limit': buy_high,
                'sell_limit': sell_low,
                'stop_limit': None,
                'qty': qty,
                'date': today,              # placement day — used for same-day idempotency
                'pred_date': signals_date,  # signals/predictions file date — for model attribution
            }
            log(f"BUY {ticker}: DAY limit {qty} shares @ ${buy_high:.2f} (${order_size:.2f}) → order {order.id}")
            placed += 1
            remaining_bp -= order_size
            save_json(OPEN_ORDERS_FILE, open_orders)  # persist immediately — prevents duplicate orders on crash
            time.sleep(ORDER_SUBMIT_DELAY)
        except Exception as e:
            log(f"[!] Buy order failed for {ticker}: {e}")

    # --- Place a profit-target sell for held positions that lack one. ---
    # The stop-loss is NOT a resting order: Alpaca permits only one resting sell
    # per fractional position, and the profit-target reserves all the shares, so
    # a companion stop-limit can never be placed. The stop is instead enforced
    # at --close (see run_close), which market-sells positions below the stop.
    for ticker, meta_entry in positions_meta.items():
        entry = open_orders.get(ticker, {})

        qty = 0.0 if dry_run else alpaca_client.get_position_qty(client, ticker)
        if qty <= 0 and not dry_run:
            continue

        sell_limit = entry.get('sell_limit') or meta_entry.get('consensus_sell_low')
        if not sell_limit or entry.get('sell_order_id'):
            continue  # no target price, or a resting sell already exists

        if dry_run:
            log(f"DRY SELL {ticker}: DAY limit {qty} shares @ ${float(sell_limit):.2f}")
            continue

        if ticker not in open_orders:
            open_orders[ticker] = {}
        try:
            order = alpaca_client.place_limit_sell(client, ticker, qty, sell_limit)
            open_orders[ticker]['sell_order_id'] = str(order.id)
            log(f"SELL {ticker}: DAY limit {qty} shares @ ${float(sell_limit):.2f} → order {order.id}")
            save_json(OPEN_ORDERS_FILE, open_orders)  # persist immediately
            time.sleep(ORDER_SUBMIT_DELAY)
        except Exception as e:
            log(f"[!] Sell order failed for {ticker}: {e}")

    print(f"\nDone. {placed} buy order(s) placed.")


# ---------------------------------------------------------------------------
# --close: settle fills and update journal
# ---------------------------------------------------------------------------

def run_close(dry_run: bool = False) -> None:
    cfg = load_trading_config()
    stop_loss_pct = float(cfg.get('stop_loss_pct', 0.95))
    max_hold_days = int(cfg.get('max_hold_days', 15))

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

    to_delete: list[str] = []
    settled: set[str] = set()  # tickers for which a fill was recorded this run

    for ticker, entry in open_orders.items():
        # Already-closed entries linger if the process crashed after record_sell
        # but before the final save. Clean them up immediately on the next run.
        if entry.get('closed'):
            to_delete.append(ticker)
            continue

        buy_oid = entry.get('buy_order_id')
        sell_oid = entry.get('sell_order_id')
        stop_oid = entry.get('stop_order_id')

        # --- Check buy order (only if no sell/stop placed yet) ---
        if buy_oid and not sell_oid and not stop_oid:
            if dry_run:
                log(f"DRY CHECK buy order {buy_oid} for {ticker}")
            else:
                order = orders_by_id.get(buy_oid)
                if order is None:
                    log(f"[!] Buy order {buy_oid} for {ticker} not found in recent orders")
                else:
                    status = str(order.status).lower().replace('orderstatus.', '')

                    if status in FILLED_STATUSES or (status in DEAD_STATUSES and _filled_qty(order) > 0):
                        fill_price = float(order.filled_avg_price or entry['buy_limit'])
                        fill_qty = _filled_qty(order, entry['qty'])
                        usd_filled = round(fill_price * fill_qty, 4)

                        # Clear and persist buy_order_id BEFORE recording the
                        # fill: record_buy is not idempotent (it accumulates
                        # into usd_invested), so if a crash/retry (e.g. from
                        # retry_failed.yml re-running this job) lands between
                        # the two steps, the durable state must already show
                        # "handled" -- otherwise the retry re-reads the same
                        # buy_order_id, sees it still filled, and double-counts
                        # the same real fill into the tracked cost basis while
                        # Alpaca's real share count doesn't change to match.
                        entry['buy_order_id'] = None
                        save_json(OPEN_ORDERS_FILE, open_orders)

                        # Use the signals/predictions date (not the placement day)
                        # so predicting_models resolves against the right file.
                        pred_date = entry.get('pred_date') or entry.get('date', today)
                        record_buy(
                            positions_meta, ticker, fill_price, usd_filled,
                            pred_date,
                            entry['buy_limit'], entry['sell_limit'],
                        )
                        log(f"BUY FILLED {ticker}: {fill_qty} shares @ ${fill_price:.2f} (${usd_filled:.2f})")

                        # P0 fix: use TOTAL held qty, not just this fill's qty.
                        # A position may have been pyramided over multiple days.
                        total_qty = alpaca_client.get_position_qty(client, ticker)
                        if total_qty <= 0:
                            log(f"[!] Could not confirm position qty for {ticker}, using fill qty")
                            total_qty = fill_qty
                        entry['qty'] = total_qty

                        # Place the profit-target DAY limit sell. The stop-loss
                        # is NOT a resting order (Alpaca allows only one resting
                        # sell per fractional position) — it's enforced by the
                        # end-of-day stop check below. Record the stop level for
                        # reference/display.
                        entry['stop_limit'] = round(entry['buy_limit'] * stop_loss_pct, 2)
                        try:
                            sell_order = alpaca_client.place_limit_sell(
                                client, ticker, total_qty, entry['sell_limit']
                            )
                            entry['sell_order_id'] = str(sell_order.id)
                            log(
                                f"SELL {ticker}: DAY limit {total_qty} shares @ "
                                f"${entry['sell_limit']:.2f} → order {sell_order.id}"
                            )
                            save_json(OPEN_ORDERS_FILE, open_orders)
                            time.sleep(ORDER_SUBMIT_DELAY)
                        except Exception as e:
                            log(f"[!] Could not place profit-target sell for {ticker}: {e}")

                    elif status in DEAD_STATUSES:
                        log(f"BUY EXPIRED {ticker}: order {buy_oid} status={status}, removing")
                        to_delete.append(ticker)
                    else:
                        log(f"{ticker}: buy order {buy_oid} still open (status={status})")

        # --- Check profit-target sell order ---
        if sell_oid and ticker not in settled:
            if dry_run:
                log(f"DRY CHECK sell order {sell_oid} for {ticker}")
            else:
                order = orders_by_id.get(sell_oid)
                if order is None:
                    # Order not in recent history — treat as expired, let --open re-place
                    log(f"[!] Sell order {sell_oid} for {ticker} not found — clearing, will re-place at open")
                    entry['sell_order_id'] = None
                else:
                    status = str(order.status).lower().replace('orderstatus.', '')

                    if status in FILLED_STATUSES or (status in DEAD_STATUSES and _filled_qty(order) > 0):
                        fill_price = float(order.filled_avg_price or entry['sell_limit'])
                        fill_qty = _filled_qty(order, entry.get('qty', 0))
                        total_qty = float(entry.get('qty') or fill_qty or 0)
                        close_fraction = _close_fraction(fill_qty, total_qty)
                        usd_returned = round(fill_price * fill_qty, 4)

                        # Cancel companion stop-loss order
                        if entry.get('stop_order_id'):
                            alpaca_client.cancel_order(client, entry['stop_order_id'])
                            log(f"  Cancelled stop-loss order {entry['stop_order_id']} for {ticker}")

                        trade = record_sell(
                            positions_meta, journal, ticker, fill_price, usd_returned, today,
                            close_fraction=close_fraction,
                        )
                        if trade:
                            log(
                                f"SELL FILLED {ticker}: {fill_qty} shares @ ${fill_price:.2f} "
                                f"→ P&L ${trade['pnl_usd']:+.4f} ({trade['pnl_pct']:+.2f}%)"
                            )
                        if close_fraction >= 0.999999:
                            entry['closed'] = True
                            to_delete.append(ticker)
                        else:
                            entry['qty'] = round(max(total_qty - fill_qty, 0.0), 6)
                            entry['sell_order_id'] = None
                            entry['stop_order_id'] = None
                            log(f"  Partial exit for {ticker}: {entry['qty']} shares remain")
                        save_json(OPEN_ORDERS_FILE, open_orders)
                        settled.add(ticker)

                    elif status in DEAD_STATUSES:
                        # DAY sell expired without filling — clear so --open re-places tomorrow
                        entry['sell_order_id'] = None
                        log(f"SELL EXPIRED {ticker}: DAY order did not fill, will re-place at open")
                    else:
                        log(f"{ticker}: sell order still pending (status={status})")

        # --- Check stop-loss sell order ---
        if stop_oid and ticker not in settled:
            if dry_run:
                log(f"DRY CHECK stop order {stop_oid} for {ticker}")
            else:
                order = orders_by_id.get(stop_oid)
                if order is None:
                    log(f"[!] Stop order {stop_oid} for {ticker} not found — clearing, will re-place at open")
                    entry['stop_order_id'] = None
                else:
                    status = str(order.status).lower().replace('orderstatus.', '')

                    if status in FILLED_STATUSES or (status in DEAD_STATUSES and _filled_qty(order) > 0):
                        fill_price = float(
                            order.filled_avg_price
                            or entry.get('stop_limit')
                            or entry.get('sell_limit', 0)
                        )
                        fill_qty = _filled_qty(order, entry.get('qty', 0))
                        total_qty = float(entry.get('qty') or fill_qty or 0)
                        close_fraction = _close_fraction(fill_qty, total_qty)
                        usd_returned = round(fill_price * fill_qty, 4)

                        # Cancel companion profit-target sell
                        if entry.get('sell_order_id'):
                            alpaca_client.cancel_order(client, entry['sell_order_id'])
                            log(f"  Cancelled profit-target order {entry['sell_order_id']} for {ticker}")

                        trade = record_sell(
                            positions_meta, journal, ticker, fill_price, usd_returned, today,
                            close_fraction=close_fraction,
                        )
                        if trade:
                            log(
                                f"STOP HIT {ticker}: {fill_qty} shares @ ${fill_price:.2f} "
                                f"→ P&L ${trade['pnl_usd']:+.4f} ({trade['pnl_pct']:+.2f}%)"
                            )
                        if close_fraction >= 0.999999:
                            entry['closed'] = True
                            to_delete.append(ticker)
                        else:
                            entry['qty'] = round(max(total_qty - fill_qty, 0.0), 6)
                            entry['sell_order_id'] = None
                            entry['stop_order_id'] = None
                            log(f"  Partial stop exit for {ticker}: {entry['qty']} shares remain")
                        save_json(OPEN_ORDERS_FILE, open_orders)
                        settled.add(ticker)

                    elif status in DEAD_STATUSES:
                        entry['stop_order_id'] = None
                        log(f"STOP EXPIRED {ticker}: DAY order did not fill, will re-place at open")
                    else:
                        log(f"{ticker}: stop order still pending (status={status})")

    # --- End-of-day stop check + max-hold-time exit -------------------------
    # Alpaca allows only one resting sell per fractional position (the
    # profit-target reserves all shares), so the stop can't be a resting order.
    # Enforce it here: any still-held position trading at/below
    # entry_price * stop_loss_pct is market-sold to cap the loss.
    #
    # Separately, a position that hits neither the profit target nor the stop
    # for max_hold_days is dead capital — it isn't confirming or denying the
    # original signal, just tying up buying power. Force-close it too so the
    # capital can be redeployed into a fresh setup.
    stopped = 0
    timed_out = 0
    if not dry_run and positions_meta:
        try:
            details = alpaca_client.get_position_details(client)
        except Exception as e:
            details = {}
            print(f"  [!] Could not fetch positions for stop check: {e}")

        today_date = datetime.strptime(today, '%Y-%m-%d').date()

        for ticker in list(positions_meta.keys()):
            if ticker in settled:
                continue  # already exited this run
            meta = positions_meta[ticker]
            pos = details.get(ticker)
            if not pos or pos['qty'] <= 0:
                continue
            price = pos['current_price']

            reason = None
            is_stop = False
            entry_price = float(meta.get('entry_price') or 0)
            if entry_price > 0:
                stop_level = round(entry_price * stop_loss_pct, 4)
                if price <= stop_level:
                    reason = f"${price:.2f} <= stop ${stop_level:.2f}"
                    is_stop = True

            if reason is None:
                entry_date_str = meta.get('entry_date')
                if entry_date_str:
                    held_days = (today_date - datetime.strptime(entry_date_str, '%Y-%m-%d').date()).days
                    if held_days >= max_hold_days:
                        reason = f"held {held_days}d >= max {max_hold_days}d"

            if reason is None:
                continue

            # Free the shares (cancel any resting profit-target sell), then
            # market-sell the whole position.
            oo = open_orders.get(ticker, {})
            if oo.get('sell_order_id'):
                alpaca_client.cancel_order(client, oo['sell_order_id'])
                oo['sell_order_id'] = None
            try:
                alpaca_client.sell_all(client, ticker)

                # Alpaca's real held qty can drift from what our state file
                # believes was bought (e.g. a duplicate buy from a workflow
                # retry that filled before the crash that triggered the
                # retry). Cap the recorded return to the tracked cost basis
                # so a desynced qty can't fabricate a P&L swing -- flag it
                # instead of inventing profit (or an inflated loss) from it.
                implied_qty = (meta.get('usd_invested', 0) / entry_price) if entry_price > 0 else pos['qty']
                qty_for_pnl = min(pos['qty'], implied_qty) if implied_qty > 0 else pos['qty']
                if abs(qty_for_pnl - pos['qty']) > 1e-6:
                    log(
                        f"  [!] {ticker}: real qty {pos['qty']:.6f} != tracked qty "
                        f"{implied_qty:.6f} -- position desynced from Alpaca; "
                        f"P&L recorded for the tracked portion only"
                    )
                usd_returned = round(price * qty_for_pnl, 4)
                trade = record_sell(positions_meta, journal, ticker, price, usd_returned, today)
                if trade:
                    label = 'EOD STOP' if is_stop else 'MAX HOLD'
                    log(
                        f"{label} {ticker}: {reason} → "
                        f"market-sold, P&L ${trade['pnl_usd']:+.4f} ({trade['pnl_pct']:+.2f}%)"
                    )
                if oo:
                    oo['closed'] = True
                to_delete.append(ticker)
                if is_stop:
                    stopped += 1
                else:
                    timed_out += 1
                time.sleep(ORDER_SUBMIT_DELAY)
            except Exception as e:
                log(f"[!] Force-sell failed for {ticker}: {e}")

    for ticker in set(to_delete):
        open_orders.pop(ticker, None)

    if not dry_run:
        save_json(OPEN_ORDERS_FILE, open_orders)

    closed = len(set(to_delete))
    print(
        f"\nDone. {closed} position(s) closed "
        f"({stopped} via end-of-day stop, {timed_out} via max-hold timeout)."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description='OracleForge DAY limit order manager.')
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
