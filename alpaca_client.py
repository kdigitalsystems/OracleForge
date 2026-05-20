# alpaca_client.py — Alpaca paper trading wrapper
"""Thin wrapper around alpaca-py for OracleForge paper trading."""
from __future__ import annotations

import os

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

KEYS_FILE = os.path.expanduser('~/.ssh/alpaca_paper_keys')


def load_keys() -> tuple[str, str, str]:
    """Parse the colon-delimited key file: Key, Secret_Key, URL."""
    if not os.path.exists(KEYS_FILE):
        raise FileNotFoundError(
            f"{KEYS_FILE} not found. "
            "Add ALPACA_PAPER_API_KEY, ALPACA_PAPER_SECRET_KEY, and ALPACA_PAPER_URL "
            "as GitHub Actions secrets (repo Settings → Secrets and variables → Actions)."
        )
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
        missing = [name for name, val in [('ALPACA_PAPER_API_KEY', api_key), ('ALPACA_PAPER_SECRET_KEY', secret_key), ('ALPACA_PAPER_URL', url)] if not val]
        raise ValueError(
            f"Alpaca keys file has empty values for: {', '.join(missing)}. "
            "Set these as GitHub Actions secrets under repo Settings → Secrets and variables → Actions."
        )
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


def buy(client: TradingClient, ticker: str, usd_amount: float) -> None:
    """Place a fractional market buy order for the given notional dollar amount."""
    order = MarketOrderRequest(
        symbol=ticker,
        notional=round(usd_amount, 2),
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
    )
    client.submit_order(order)


def sell_all(client: TradingClient, ticker: str) -> None:
    """Close the entire position for a ticker at market."""
    client.close_position(ticker)
