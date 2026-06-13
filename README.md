# OracleForge

An automated paper-trading assistant that uses a **local LLM ensemble** to generate buy/sell price ranges overnight, then places limit orders on Alpaca at market open and settles them at market close.

> Research / paper trading only. Not financial advice.

---

## Dashboard

Live results are published automatically to GitHub Pages after every nightly run and every market close:  
**https://kdigitalsystems.github.io/OracleForge**

| Tab | Content |
|---|---|
| **Signals** | Today's ACTIVE/SKIP/STALE setups with consensus buy/sell ranges, upside chart, and model disagreement (CV) |
| **P&L** | Cumulative P&L curve, win rate, best/worst trades, full trade journal |
| **Backtest** | Historical simulation with profit factor, avg win/loss %, and max consecutive losses |
| **Model Scores** | Current ensemble weights (0–10 scale) per model, updated nightly with recency decay |

The page has a **Rebuild** button that re-generates the dashboard on demand (requires a GitHub PAT with `repo` scope stored in your browser's local storage).

---

## Trading logic

### Overnight (forge_loop.py)

Each night, local Ollama models independently analyse every ticker on the watchlist:

1. **Fetch market data** — latest daily OHLC bars pulled from Alpaca for all tickers in one batch.
2. **Evaluate prior predictions** — compare yesterday's predicted buy/sell ranges against today's realized prices:
   - OHLC check: did price touch the buy range? Did it then reach the sell range (win), or breach `buy_high × 0.95` (stop)? → ±0.01 score delta per model.
   - Trade journal check: for any positions closed yesterday, contributing models are adjusted by actual realized P&L (`pnl_pct × trade_score_scale`, capped by `trade_score_cap`). By default, a +1% trade gives +0.02, while a −5% stopped trade gives −0.10.
   - **Fallback predictions are excluded** from both consensus and scoring — they are tagged `fallback: true` and skipped.
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
4. **Consensus** — predictions are combined via a score-weighted average. Models with better historical accuracy carry more weight. At least 2 models must agree for a signal to be emitted. A **model disagreement gate** (CV > 10% on `buy_high` or `sell_low`) suppresses signals where models are too far apart.
5. **Signal classification**:
   - **ACTIVE** — consensus buy range is reachable with >1% upside to sell range. Valid setup.
   - **SKIP** — setup exists but upside is too small or buy range is too wide.
   - **STALE** — price has already moved above the sell range (missed opportunity). `upside_pct` is `null` for stale signals.
6. **Technical context** provided to each model: a table of the **last ~10 daily OHLC+volume bars** (so the model sees real recent price action, not just summary stats), plus close price, RSI(14) via Wilder's smoothing, volume ratio, 10d high/low distance, SMA20 distance, Bollinger Band %B, and 5-day price momentum.
7. **Score decay** — model scores decay by 0.99× per day before each update, so recent performance carries more weight than distant history.
8. **Persist** — saves enriched predictions to `history/predictions_YYYY-MM-DD.json`, signals report to `reports/signals_YYYY-MM-DD.json`, updated model scores to `state/analyst_scores.json`.

> **Staged & resumable:** a full run is long (one LLM call per model per ticker; ~5h for ~150 tickers × 3 models — that total is inherent to serial local-LLM inference and staging chunks it, it doesn't parallelize it). `forge_loop.py` checkpoints predictions/signals to disk every `--batch-size` tickers (default 25; resume granularity) and, with `--push`, commits + pushes every `--push-every` tickers (a "stage"). The CI nightly uses `--batch-size 10 --push-every 100`: it saves every 10 tickers but pushes after ~100, so it processes ~100 tickers, pushes, processes the next ~100, and so on — few commits, yet a crash only re-runs the unfinished tickers. The model-score update is computed once per day (first batch); resumed invocations skip it.

### Daytime (trader.py — two short jobs, no polling)

**Morning (`trader.py --open`, 9:30 AM ET):**
```
For each ACTIVE ticker:
  if no existing position AND no order placed today:
    qty = min($2, $8 - existing_position) / buy_high
    place DAY limit buy @ buy_high

For each existing position without a sell order:
  place DAY limit sell @ consensus sell_low
  place DAY stop-limit sell with stop @ buy_high × 0.95
```

**Evening (`trader.py --close`, 4:05 PM ET):**
```
For each tracked buy order:
  if FILLED  → record entry price; place DAY sell @ sell_low + stop-limit @ buy_high × 0.95
  if EXPIRED → remove from order state

For each tracked sell order:
  if FILLED  → cancel companion stop-loss; record P&L; mark closed
  if EXPIRED → clear sell_order_id so --open re-places it next morning

For each tracked stop-loss order:
  if FILLED  → cancel companion sell; record P&L as loss; mark closed
  if EXPIRED → clear stop_order_id so --open re-places it next morning
```

Alpaca handles execution during the day. No process stays alive between the two jobs.

> **Why DAY, not GTC?** Alpaca does not support GTC orders for fractional share quantities.
> DAY sell and stop orders are re-placed every morning until the position is exited.

**Position limits** (configurable in `config/trading.json`):
- Max $2 per order
- Max $8 total position per ticker
- Fractional shares via Alpaca limit orders (qty-based)
- Stop-loss at `buy_high × stop_loss_pct` (default 0.95 = 5% below entry)

### Feedback loop

Every closed trade feeds back into the next night's run:
- Model scores update based on actual P&L percentage, not just win/loss labels. Real fills are weighted higher than theoretical OHLC checks, capped to avoid overreacting to one outlier, and decayed by 0.99× daily to emphasise recent accuracy.
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
Keys are never stored in GitHub Secrets.

| Workflow | Schedule | What it does |
|---|---|---|
| [Nightly Forge](.github/workflows/nightly_forge.yml) | 23:00 UTC weekdays | Runs unit tests → `update_tickers.py` → `forge_loop.py` → validates outputs → refreshes recent walk-forward study → regenerates dashboard → commits state |
| [Morning Orders](.github/workflows/morning_orders.yml) | 13:30 UTC weekdays (9:30 AM ET) | Places DAY limit buy orders for ACTIVE tickers; re-places DAY sell + stop-loss orders for held positions |
| [Evening Cleanup](.github/workflows/evening_cleanup.yml) | 20:05 UTC weekdays (4:05 PM ET) | Detects fills, records P&L, cancels companion orders, clears expired orders, refreshes trade attribution, regenerates dashboard |
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
| `python3 forge_loop.py --batch-size 10 --push-every 100 --push` | Checkpoint every 10 tickers, push every 100 (CI staging) |
| `python3 trader.py --open` | Place DAY limit buy orders at market open |
| `python3 trader.py --close` | Settle fills and update P&L journal |
| `python3 trader.py --open --dry-run` | Preview orders without placing them |
| `python3 trader.py --close --dry-run` | Preview settlement without writing state |
| `python3 backtest.py` | Score historical predictions vs realized OHLC |
| `python3 backtest.py --from-date 2026-05-01 --to-date 2026-05-15` | Backtest a bounded date window |
| `python3 backtest.py --max-dates 5` | Backtest only the most recent 5 prediction dates |
| `python3 backtest.py --compare-executions` | A/B the ACTIVE edge under limit_stop / limit_hold / market_hold execution models |
| `python3 backtest.py --walk-forward` | Train parameters on rolling prior windows, validate on unseen future windows |
| `python3 backtest.py --attribution` | Summarise actual closed-trade attribution by model, ticker, and execution quality |
| `python3 scripts/generate_html_report.py` | Regenerate `docs/index.html` from local data |

---

## Backtesting

Walk the saved prediction history and score each predicted buy/sell range against the
**next trading session's** realized OHLC (fetched from yfinance, independent of Alpaca).

```bash
# Backtest all available prediction history
python3 backtest.py

# Restrict to a date range (both inclusive, YYYY-MM-DD)
python3 backtest.py --from-date 2026-05-01
python3 backtest.py --from-date 2026-05-01 --to-date 2026-05-15
```

Each `history/predictions_YYYY-MM-DD.json` file is replayed: entry is assumed at `buy_high`
(conservative), with the stop at `buy_high × stop_loss_pct` — the **same** `stop_loss_pct`
(default 0.95 = −5%) used by the live trader and the nightly scorer, read from
`config/trading.json` so the backtest matches what is actually traded. Outcomes are bucketed as
**win** (price reached `sell_low`), **stop** (price fell to the stop), **miss** (triggered but
neither target nor stop hit), or **no_trigger**. Win/loss averages and profit factor are
aggregated by the sign of each trade's realized return.

**Output:**
- **Console summary** — per-model and per-signal tables: trade count, win %, avg win %,
  avg loss %, profit factor, and max consecutive losses.
- **Significance read** — each signal bucket reports whether its mean per-trade return is
  statistically distinguishable from zero, and whether the sample is even large enough
  (`>= 30` trades) to trust. Guards against tuning on noise.
- **Benchmark vs. buy-and-hold** — for ACTIVE names it compares the strategy return against
  simply buying the same names at the prior close and holding to the next close (paired,
  with significance), plus the whole-universe drift. A one-line **verdict** states whether the
  signal layer is actually adding value, hurting, or indistinguishable from the baseline.
- **`reports/backtest_summary.json`** — full report (daily equity curve, significance,
  benchmark), which the dashboard's **Backtest** tab renders. A timestamped snapshot is also
  archived to `reports/backtest_history/` so metric evolution is trackable over time.
- **`reports/walk_forward_summary.json`** — rolling train/test parameter study. The optimizer
  selects the best parameter set on prior prediction dates, then measures it on later unseen
  dates. This is evidence-only: it recommends settings but never rewrites live config.
- **`reports/trade_attribution.json`** — actual closed-trade breakdown by model and ticker,
  plus execution-quality diagnostics such as average entry improvement versus `buy_high` and
  target capture versus the consensus sell range.

```bash
# Pick parameters on rolling 4-date windows, validate on the next date
python3 backtest.py --walk-forward

# Use wider windows once there is more history
python3 backtest.py --walk-forward --train-window 10 --test-window 2

# Fast recent-history check when yfinance is slow
python3 backtest.py --walk-forward --max-dates 5 --train-window 3 --test-window 1

# Attribute actual closed trades
python3 backtest.py --attribution
```

**Requirements:**
- At least one completed nightly run (so `history/predictions_*.json` exists). With no history,
  the command prints `No prediction history files found in history/.` and exits.
- Outbound internet for yfinance. Tickers with no available next-session bar (too recent,
  delisted, etc.) are counted under `skipped_pairs` rather than failing the run.

---

## Configuration

| File | Purpose | Key fields |
|---|---|---|
| `config/tickers.json` | Active watchlist (built by `update_tickers.py`) | — |
| `config/universe.json` | Ticker filter thresholds | `min_price`, `min_avg_daily_volume`, `max_daily_volatility_pct`, `max_tickers` |
| `config/trading.json` | Position limits and risk settings | `max_per_trade_usd` (2.0), `max_position_usd` (8.0), `stop_loss_pct` (0.95), `score_decay_per_day` (0.99), `trade_score_scale` (0.02), `trade_score_cap` (0.10) |
| `config/signals.json` | Signal classification thresholds | `min_upside_pct` (1.5), `max_spread_pct` (3.0), `min_agreeing_models` (2), `max_consensus_cv` (0.10) |
| `config/models.json` | Model list (Ollama model IDs) | Never overwritten by automation |
| `state/analyst_scores.json` | Model scores — add/remove models here | 0.0–10.0 per model, initialised at 5.0 |

### config/trading.json

```json
{
    "max_per_trade_usd": 2.0,
    "max_position_usd": 8.0,
    "stop_loss_pct": 0.95,
    "score_decay_per_day": 0.99,
    "trade_score_scale": 0.02,
    "trade_score_cap": 0.10
}
```

### config/signals.json

```json
{
    "min_upside_pct": 1.5,
    "max_spread_pct": 3.0,
    "min_bullish_models": 2,
    "min_agreeing_models": 2,
    "max_consensus_cv": 0.10
}
```

---

## State files

### state/open_orders.json

Bridges the morning `--open` job and the evening `--close` job. Schema per ticker:

```json
{
  "NVDA": {
    "buy_order_id": "abc123",
    "sell_order_id": "def456",
    "stop_order_id": "ghi789",
    "buy_limit": 890.50,
    "sell_limit": 910.00,
    "stop_limit": 846.00,
    "qty": 0.00225,
    "date": "2025-05-20",
    "closed": false
  }
}
```

- `sell_order_id` and `stop_order_id` are companion DAY orders; whichever fills first cancels the other.
- `closed: true` is written immediately after `record_sell` to prevent double-recording on crash/retry.

### state/open_positions_meta.json

Tracks open position entry data for P&L calculation:

```json
{
  "NVDA": {
    "entry_price": 890.50,
    "usd_invested": 4.00,
    "date": "2025-05-19",
    "buy_limit": 890.50,
    "sell_limit": 910.00
  }
}
```

---

## Project layout

| Path | Purpose |
|---|---|
| `forge_loop.py` | Overnight inference engine |
| `trader.py` | Morning open + evening close order jobs |
| `signals.py` | Consensus scoring, CV gate, signal classification |
| `alpaca_client.py` | Alpaca API wrapper (keys from `~/.ssh/alpaca_paper_keys`) |
| `update_tickers.py` | Dynamic universe builder from Alpaca assets |
| `backtest.py` | Historical simulation against realized OHLC |
| `scripts/generate_html_report.py` | Generates `docs/index.html` static dashboard |
| `scripts/validate_outputs.py` | Post-run sanity checks for nightly CI |
| `docs/index.html` | Published GitHub Pages dashboard (auto-generated) |
| `history/predictions_*.json` | Per-model daily predictions with consensus and CV |
| `history/trade_journal.json` | Cumulative closed trade P&L |
| `reports/signals_*.json` | Daily ACTIVE/SKIP/STALE signal reports |
| `state/open_orders.json` | Live order state (buy, sell, stop) shared between --open and --close jobs |
| `state/open_positions_meta.json` | Open position entry tracking (entry price, USD invested) |
| `state/analyst_scores.json` | Model credibility weights (updated nightly with recency decay) |

---

## Prerequisites

- Python 3.12+
- [Ollama](https://ollama.com/) running at `http://localhost:11434`
- Alpaca paper trading account; keys at `~/.ssh/alpaca_paper_keys`
- Self-hosted GitHub Actions runner registered to this repo (for automation)

---

## Tests

```bash
python3 -m unittest test_signals test_backtest test_forge test_trader test_alpaca_client -v
```

**115 tests** covering:

| Module | Areas |
|---|---|
| `test_signals` | Consensus weighting, CV disagreement gate, fallback exclusion, signal classification, STALE upside handling |
| `test_backtest` | Win/stop/miss simulation, profit factor, avg win/loss %, max consecutive losses, internal field cleanup |
| `test_forge` | LLM output parsing, fallback tagging, RSI/Bollinger/%B/momentum technicals, score delta feedback, recency decay, stop-threshold evaluation |
| `test_trader` | `record_buy` (new + averaging), `record_sell` (win/loss/missing), P&L calculation, trade provenance fields |

---

## License

MIT — see [LICENSE](LICENSE).
