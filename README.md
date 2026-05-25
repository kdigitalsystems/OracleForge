# OracleForge

An automated paper-trading assistant that uses a **local LLM ensemble** to generate buy/sell price ranges overnight, then places limit orders on Alpaca at market open and settles them at market close.

> Research / paper trading only. Not financial advice.

---

## Dashboard

Live results are published automatically to GitHub Pages after every nightly run and every market close:  
**https://kdigitalsystems.github.io/OracleForge**

| Tab | Content |
|---|---|
| **Signals** | Today's ACTIVE/SKIP/STALE setups with consensus buy/sell ranges and upside chart |
| **P&L** | Cumulative P&L curve, win rate, best/worst trades, full trade journal |
| **Backtest** | Historical simulation of predicted ranges vs. realized OHLC |
| **Model Scores** | Current ensemble weights (0–10 scale) per model |

The page has a **Rebuild** button that re-generates the dashboard on demand (requires a GitHub PAT with `repo` scope stored in your browser's local storage).

---

## Trading logic

### Overnight (forge_loop.py)

Each night, local Ollama models independently analyse every ticker on the watchlist:

1. **Fetch market data** — latest daily OHLC bars pulled from Alpaca for all tickers in one batch.
2. **Evaluate prior predictions** — compare yesterday's predicted buy/sell ranges against today's realized prices:
   - OHLC check: did price touch the buy range? Did it then reach the sell range? → ±0.01 score delta per model.
   - Trade journal check: for any positions closed yesterday, models that predicted the winning setup get +0.02; losers get −0.02.
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
4. **Consensus** — predictions are combined via a score-weighted average. Models with better historical accuracy carry more weight. At least 2 models must agree for a signal to be emitted.
5. **Signal classification**:
   - **ACTIVE** — consensus buy range is reachable with >1% upside to sell range. Valid setup.
   - **SKIP** — setup exists but upside is too small or buy range is too wide.
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
  place DAY limit sell @ consensus sell_low
```

**Evening (`trader.py --close`, 4:05 PM ET):**
```
For each tracked buy order:
  if FILLED  → record entry price, place DAY sell @ sell_low
  if EXPIRED → remove from order state

For each tracked sell order:
  if FILLED  → record P&L to history/trade_journal.json, remove from state
  if EXPIRED → clear sell_order_id so --open re-places it next morning
```

Alpaca handles execution during the day. No process stays alive between the two jobs.

> **Why DAY, not GTC?** Alpaca does not support GTC orders for fractional share quantities.
> DAY sell orders are re-placed every morning until the position is exited.

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
python3 update_tickers.py

# Run overnight analysis (or test with a small list)
python3 forge_loop.py --tickers NVDA,AAPL

# Dry-run the trading jobs (logs without placing or recording anything)
python3 trader.py --open --dry-run
python3 trader.py --close --dry-run
```

---

## Automation (GitHub Actions)

All four workflows run on a **self-hosted runner** and read Alpaca keys directly from
`~/.ssh/alpaca_paper_keys` (colon-delimited: `Key:`, `Secret_Key:`, `URL:`).

| Workflow | Schedule | What it does |
|---|---|---|
| [Nightly Forge](.github/workflows/nightly_forge.yml) | 23:00 UTC weekdays | Runs unit tests → `update_tickers.py` → `forge_loop.py` → regenerates dashboard → commits state |
| [Morning Orders](.github/workflows/morning_orders.yml) | 13:30 UTC weekdays (9:30 AM ET) | Places DAY limit buy orders for ACTIVE tickers; re-places DAY sell orders for held positions |
| [Evening Cleanup](.github/workflows/evening_cleanup.yml) | 20:05 UTC weekdays (4:05 PM ET) | Detects fills, records P&L, clears expired orders, regenerates dashboard |
| [Rebuild Dashboard](.github/workflows/regenerate_report.yml) | Manual (via Rebuild button) | Regenerates `docs/index.html` from existing data files and commits |

### Runner registration

The self-hosted runner must be registered to this repository. To register:

1. Go to **Settings → Actions → Runners → New self-hosted runner** on GitHub.
2. Copy the registration token.
3. On the runner machine (WSL):
   ```bash
   cd ~/github/actions-runner
   ./config.sh --url https://github.com/kdigitalsystems/OracleForge --token <TOKEN> --replace
   ```

---

## Commands

| Command | Purpose |
|---|---|
| `python3 update_tickers.py` | Rebuild watchlist from Alpaca universe |
| `python3 update_tickers.py --limit 50 --min-price 20 --max-vol 3.0` | Custom filters |
| `python3 forge_loop.py` | Overnight analysis for all tickers |
| `python3 forge_loop.py --tickers NVDA,AAPL` | Selected tickers only |
| `python3 trader.py --open` | Place DAY limit buy orders at market open |
| `python3 trader.py --close` | Settle fills and update P&L journal |
| `python3 trader.py --open --dry-run` | Preview orders without placing them |
| `python3 trader.py --close --dry-run` | Preview settlement without writing state |
| `python3 backtest.py` | Score historical predictions vs realized OHLC |
| `python3 scripts/generate_html_report.py` | Regenerate `docs/index.html` from local data |

---

## Configuration

| File | Purpose |
|---|---|
| `config/tickers.json` | Active watchlist (built by `update_tickers.py`) |
| `config/universe.json` | Ticker filter thresholds (price, volume, volatility, max count) |
| `config/trading.json` | Position limits (`max_per_trade_usd`, `max_position_usd`) |
| `config/signals.json` | Signal classification thresholds (`min_upside_pct`, `max_spread_pct`) |
| `state/analyst_scores.json` | Model scores — add/remove models here |

---

## Project layout

| Path | Purpose |
|---|---|
| `forge_loop.py` | Overnight inference engine |
| `trader.py` | Morning open + evening close order jobs |
| `signals.py` | Consensus scoring and signal classification |
| `alpaca_client.py` | Alpaca API wrapper (keys from `~/.ssh/alpaca_paper_keys`) |
| `update_tickers.py` | Dynamic universe builder from Alpaca assets |
| `backtest.py` | Historical simulation against realized OHLC |
| `scripts/generate_html_report.py` | Generates `docs/index.html` static dashboard |
| `scripts/validate_outputs.py` | Post-run sanity checks for nightly CI |
| `docs/index.html` | Published GitHub Pages dashboard (auto-generated) |
| `history/predictions_*.json` | Per-model daily predictions with consensus |
| `history/trade_journal.json` | Cumulative closed trade P&L |
| `reports/signals_*.json` | Daily ACTIVE/SKIP/STALE signal reports |
| `state/open_orders.json` | Live order state shared between --open and --close jobs |
| `state/open_positions_meta.json` | Open position entry tracking (entry price, USD invested) |
| `state/analyst_scores.json` | Model credibility weights (updated nightly) |

---

## Prerequisites

- Python 3.12+
- [Ollama](https://ollama.com/) running at `http://localhost:11434`
- Alpaca paper trading account; keys at `~/.ssh/alpaca_paper_keys`
- Self-hosted GitHub Actions runner registered to this repo (for automation)

---

## Tests

```bash
python3 -m unittest test_signals test_backtest test_forge test_trader -v
```

58 tests covering: signal consensus, classification, backtest simulation, LLM output parsing, score feedback, and trade recording (record_buy / record_sell).

---

## License

MIT — see [LICENSE](LICENSE).
