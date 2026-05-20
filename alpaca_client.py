# alpaca_client.py — Alpaca paper trading wrapper
"""Thin wrapper around alpaca-py for OracleForge paper trading."""
from __future__ import annotations

import os

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

KEYS_FILE = os.path.expanduser('~/.ssh/alpaca_paper_keys')


def load_keys() -> tuple[str, str, str]:
    """Parse the colon-delimited key file: Key, Secret_Key, URL."""
    if not os.path.exists(KEYS_FILE):
        raise FileNotFoundError(f"{KEYS_FILE} not found.")
    keys: dict[str, str] = {}
    with open(KEYS_FILE) as f:
        for line in f:
            line = line.strip()
            if ':' in line:
                k, v = line.split(':', 1)
                keys[k.strip()] = v.strip()
    api_key = keys.get('Key', '')
    secret_key = keys.get('Secret_Key', '')
    url = keys.get('URL', '')
    if not api_key or not secret_key or not url:
        missing = [n for n, v in [('Key', api_key), ('Secret_Key', secret_key), ('URL', url)] if not v]
        raise ValueError(f"Alpaca keys file missing values for: {', '.join(missing)}")
    return api_key, secret_key, url


def get_trading_client() -> TradingClient:
    api_key, secret_key, base_url = load_keys()
    return TradingClient(api_key, secret_key, url_override=base_url)


def get_data_client() -> StockHistoricalDataClient:
    api_key, secret_key, _ = load_keys()
    return StockHistoricalDataClient(api_key, secret_key)


def get_positions(client: TradingClient) -> dict[str, float]:
    """Return {symbol: market_value_usd} for all open positions."""
    return {p.symbol: float(p.market_value) for p in client.get_all_positions()}


def get_position_qty(client: TradingClient, ticker: str) -> float:
    """Return held share quantity for ticker, or 0.0 if no position."""
    try:
        pos = client.get_open_position(ticker)
        return float(pos.qty)
    except Exception:
        return 0.0


def place_limit_buy(client: TradingClient, ticker: str, qty: float,
                    limit_price: float, time_in_force: str = 'day'):
    """Place a fractional limit buy order. Returns the order object."""
    tif = TimeInForce.DAY if time_in_force == 'day' else TimeInForce.GTC
    req = LimitOrderRequest(
        symbol=ticker,
        qty=round(qty, 6),
        side=OrderSide.BUY,
        time_in_force=tif,
        limit_price=round(limit_price, 2),
    )
    return client.submit_order(req)


def place_limit_sell(client: TradingClient, ticker: str, qty: float, limit_price: float):
    """Place a GTC fractional limit sell order. Returns the order object."""
    req = LimitOrderRequest(
        symbol=ticker,
        qty=round(qty, 6),
        side=OrderSide.SELL,
        time_in_force=TimeInForce.GTC,
        limit_price=round(limit_price, 2),
    )
    return client.submit_order(req)


def get_order(client: TradingClient, order_id: str):
    """Fetch a single order by ID."""
    return client.get_order_by_id(order_id)


def cancel_order(client: TradingClient, order_id: str) -> None:
    """Cancel an order, ignoring errors if already filled or cancelled."""
    try:
        client.cancel_order_by_id(order_id)
    except Exception:
        pass


def buy(client: TradingClient, ticker: str, usd_amount: float) -> None:
    """Place a fractional market buy (notional). Used for dry-run compatibility."""
    req = MarketOrderRequest(
        symbol=ticker,
        notional=round(usd_amount, 2),
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
    )
    client.submit_order(req)


def sell_all(client: TradingClient, ticker: str) -> None:
    """Close the entire position for a ticker at market."""
    client.close_position(ticker)
