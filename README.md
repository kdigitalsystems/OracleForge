# OracleForge

An automated paper-trading assistant that uses a **local LLM ensemble** to generate buy/sell price ranges overnight, then monitors live prices during market hours and executes trades on Alpaca when conditions are met.

> Research / paper trading only. Not financial advice.

---

## Trading logic

### Overnight (forge_loop.py)

Each night, three local Ollama models independently analyse every ticker on the watchlist:

1. **Fetch market data** — latest daily OHLC bars pulled from Alpaca for all tickers in one batch.
2. **Evaluate prior predictions** — compare yesterday's predicted buy/sell ranges against today's realized prices:
   - OHLC check: did price touch the buy range? Did it then reach the sell range? → ±0.01 score delta per model.
   - Trade journal check: for any positions actually closed yesterday, models that predicted the winning setup get +0.02; losers get −0.02.
3. **AI inference** — each model receives the closing price, recent news headlines, its own historical win rate, and the last 5 actual trade results for that ticker. It responds with:
   ```json
   {
     "buy_low": 178.50,
     "buy_high": 180.00,
     "sell_low": 185.00,
     "sell_high": 187.50,
     "rationale": "..."
   }
   ```
4. **Consensus** — model predictions are combined via a score-weighted average. Models with better historical accuracy carry more weight.
5. **Signal classification**:
   - **ACTIVE** — consensus buy range is below current price with >1% upside to sell range. Valid setup.
   - **SKIP** — setup exists but upside is too small or spread is too wide.
   - **STALE** — price has already moved above the sell range (missed opportunity).
6. **Persist** — saves enriched predictions to `history/predictions_YYYY-MM-DD.json`, signals report to `reports/signals_YYYY-MM-DD.json`, updated model scores to `state/analyst_scores.json`.

### Daytime (trader.py — two short jobs, no polling)

**Morning (`trader.py --open`, 9:30 AM ET):**
```
For each ACTIVE ticker:
  if no position cap breached AND no order placed today:
    qty = min($2, $8 - existing_position) / buy_high
    place DAY limit buy @ buy_high

For each existing position without a sell order:
  place GTC limit sell @ consensus sell_low
```

**Evening (`trader.py --close`, 4:05 PM ET):**
```
For each tracked buy order:
  if FILLED  → record entry price, place GTC sell @ sell_low
  if EXPIRED → remove from order state

For each tracked sell order:
  if FILLED  → record P&L to history/trade_journal.json
  if OPEN    → leave as GTC (carries to next session)
```

Alpaca handles execution during the day. No process stays alive.

**Position limits** (configurable in `config/trading.json`):
- Max $2 per order
- Max $8 total position per ticker
- Fractional shares via Alpaca limit orders (qty-based)

**Stop-loss** (built into scoring): if the low of day breaches `buy_high × 0.98`, the prediction is scored as a loss and the model's score is penalised.

### Feedback loop

Every closed trade feeds back into the next night's run:
- Model scores update based on actual P&L outcomes (weighted higher than theoretical OHLC checks).
- Each model's LLM prompt includes its own win rate and recent P&L, so it receives direct feedback on its performance.
- Over time, the consensus naturally shifts toward models that make accurate predictions.

---

## Quick start

```bash
pip install -r requirements.txt

# Pull Ollama models
ollama pull llama3.1:8b-instruct-q8_0
ollama pull qwen2.5:14b-instruct-q4_K_M
ollama pull deepseek-r1:8b

# Build ticker watchlist (top 200 liquid, low-volatility US equities)
python update_tickers.py

# Run overnight analysis (or test with a small list)
python forge_loop.py --tickers NVDA,AAPL

# Run daytime trader (dry run — logs without placing orders)
python trader.py --dry-run

# Dashboard
streamlit run dashboard.py
```

---

## Automation (GitHub Actions)

| Workflow | Schedule | What it does |
|---|---|---|
| [Nightly Forge](.github/workflows/nightly_forge.yml) | 23:00 UTC weekdays | `update_tickers.py` → `forge_loop.py` → commit state |
| [Morning Orders](.github/workflows/morning_orders.yml) | 13:30 UTC weekdays (9:30 AM ET) | Places DAY limit buy orders + GTC sell orders for held positions (~30 sec) |
| [Evening Cleanup](.github/workflows/evening_cleanup.yml) | 20:05 UTC weekdays (4:05 PM ET) | Detects fills, records P&L to journal, cancels expired orders (~30 sec) |

Both workflows run on a self-hosted runner and read Alpaca keys directly from `~/.ssh/alpaca_paper_keys` (colon-delimited: `Key:`, `Secret_Key:`, `URL:`).

---

## Commands

| Command | Purpose |
|---|---|
| `python update_tickers.py` | Rebuild watchlist from Alpaca universe |
| `python update_tickers.py --limit 50 --min-price 20 --max-vol 3.0` | Custom filters |
| `python forge_loop.py` | Overnight analysis for all tickers |
| `python forge_loop.py --tickers NVDA,AAPL` | Watchlist mode — selected tickers only |
| `python trader.py --open` | Place limit buy orders at market open |
| `python trader.py --close` | Settle fills and update P&L journal |
| `python trader.py --open --dry-run` | Preview orders without placing them |
| `python trader.py --close --dry-run` | Preview settlement without writing state |
| `python backtest.py` | Score historical predictions vs realized OHLC |
| `streamlit run dashboard.py` | Web UI |

---

## Dashboard tabs

| Tab | Content |
|---|---|
| **Signals** | Today's ACTIVE/SKIP/STALE setups with consensus buy/sell ranges and per-model breakdown |
| **Positions** | Live Alpaca paper positions and portfolio value |
| **Trades** | Today's buy/sell log with prices and amounts |
| **P&L** | Cumulative P&L chart, win rate by model, best/worst trades, full journal |
| **Model scores** | Current ensemble weights (0–10 scale) |

---

## Configuration

| File | Purpose |
|---|---|
| `config/tickers.json` | Active watchlist (built by `update_tickers.py`) |
| `config/universe.json` | Ticker filter thresholds (price, volume, volatility, max count) |
| `config/trading.json` | Position limits and poll interval |
| `config/signals.json` | Signal classification thresholds (min upside, max spread) |
| `state/analyst_scores.json` | Model scores — add/remove models here |

---

## Project layout

| Path | Purpose |
|---|---|
| `forge_loop.py` | Overnight inference engine |
| `trader.py` | Daytime price monitor and order executor |
| `signals.py` | Consensus scoring and signal classification |
| `alpaca_client.py` | Alpaca API wrapper (keys from `~/.ssh/alpaca_paper_keys`) |
| `update_tickers.py` | Dynamic universe builder from Alpaca assets |
| `backtest.py` | Historical simulation against realized OHLC |
| `dashboard.py` | Streamlit UI |
| `history/predictions_*.json` | Per-model daily predictions with consensus |
| `history/trade_journal.json` | Cumulative closed trade P&L |
| `reports/signals_*.json` | Daily ACTIVE/SKIP/STALE signal reports |
| `reports/trades_*.json` | Intraday trade logs |
| `state/analyst_scores.json` | Model credibility weights (updated nightly) |
| `state/open_positions_meta.json` | Open position entry tracking (entry price, USD invested) |

---

## Prerequisites

- Python 3.12+
- [Ollama](https://ollama.com/) running at `http://localhost:11434`
- Alpaca paper trading account; keys at `~/.ssh/alpaca_paper_keys`
- Self-hosted GitHub Actions runner (for automation)

## Tests

```bash
python -m unittest test_signals.py test_backtest.py
```

## License

MIT — see [LICENSE](LICENSE).
