"""Quick status check — prints account, open orders, and recent closed orders."""
import alpaca_client
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

client = alpaca_client.get_trading_client()

# Account
account = client.get_account()
print(f"=== Account ===")
print(f"  Portfolio value : ${float(account.portfolio_value):,.2f}")
print(f"  Equity          : ${float(account.equity):,.2f}")
print(f"  Buying power    : ${float(account.buying_power):,.2f}")

# Positions
positions = client.get_all_positions()
print(f"\n=== Open Positions ({len(positions)}) ===")
for p in sorted(positions, key=lambda x: float(x.unrealized_pl or 0), reverse=True)[:15]:
    unreal = float(p.unrealized_pl or 0)
    pct = float(p.unrealized_plpc or 0) * 100
    print(f"  {p.symbol:8} qty={p.qty:10} entry=${float(p.avg_entry_price):8.3f}  "
          f"mkt=${float(p.current_price):8.3f}  P&L ${unreal:+.4f} ({pct:+.2f}%)")
all_unreal = sum(float(p.unrealized_pl or 0) for p in positions)
print(f"  ... ({len(positions)} total positions)  Total unrealized P&L: ${all_unreal:+.4f}")

# Open orders on Alpaca
open_orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=100))
print(f"\n=== Open Orders on Alpaca ({len(open_orders)}) ===")
buys  = [o for o in open_orders if 'buy'  in str(o.side).lower()]
sells = [o for o in open_orders if 'sell' in str(o.side).lower()]
print(f"  Buy orders : {len(buys)}")
print(f"  Sell orders: {len(sells)}")
for o in sells[:10]:
    print(f"  SELL {o.symbol:8} qty={o.qty} @ ${o.limit_price} [{o.status}]")

# Recent fills
closed = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=30))
filled = [o for o in closed if 'filled' in str(o.status).lower()]
print(f"\n=== Recent Filled Orders ({len(filled)}) ===")
for o in filled[:15]:
    print(f"  {str(o.side):20} {o.symbol:8} qty={o.filled_qty:10} "
          f"@ ${float(o.filled_avg_price or 0):8.3f}  [{o.filled_at}]")
