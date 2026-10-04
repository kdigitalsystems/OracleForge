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
| **Backtest** | Replay of stored signals under the live trading rules: returns by exit, benchmark, per-model results, calibration against the live journal, and the walk-forward study |
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

> **Staged & resumable:** a full run is long (one LLM call per model per ticker; hours for ~150 tickers × N models in `config/models.json` — that total is inherent to serial local-LLM inference and staging chunks it, it doesn't parallelize it). `forge_loop.py` checkpoints predictions/signals to disk every `--batch-size` tickers (default 25; resume granularity) and, with `--push`, commits + pushes every `--push-every` tickers (a "stage"). The CI nightly uses `--batch-size 10 --push-every 100`: it saves every 10 tickers but pushes after ~100, so it processes ~100 tickers, pushes, processes the next ~100, and so on — few commits, yet a crash only re-runs the unfinished tickers. The model-score update is computed once per day (first batch); resumed invocations skip it.

### Daytime (trader.py — two short jobs, no polling)

**Morning (`trader.py --open`, 9:30 AM EDT / 8:30 AM EST):**
```
For each ACTIVE ticker:
  if no existing position AND no order placed today:
    qty = min($2, $8 - existing_position) / buy_high
    place DAY limit buy @ buy_high

For each existing position without a resting sell:
  place DAY limit sell @ consensus sell_low   (single resting profit-target order)
```

**Evening (`trader.py --close`, 5:05 PM EDT / 4:05 PM EST):**
```
For each tracked buy order:
  if FILLED  → record entry price; place DAY profit-target sell @ sell_low
  if EXPIRED → remove from order state

For each tracked sell order:
  if FILLED  → record P&L (full or partial); mark closed
  if EXPIRED → clear sell_order_id so --open re-places it next morning

End-of-day stop check (all still-held positions):
  if current price ≤ entry_price × stop_loss_pct → market-sell, record the loss
  elif held ≥ max_hold_days without hitting target or stop → market-sell (stale position)
```

Alpaca handles execution during the day. No process stays alive between the two jobs.

> **Why one resting order + an EOD stop?** Alpaca supports neither GTC nor
> bracket/OCO orders for *fractional* shares, and only **one resting sell** can
> reserve a position's shares at a time. So the profit-target limit is the
> single resting order, and the stop-loss is enforced at `--close`: any
> position trading at/below its stop level is market-sold (a DAY limit profit
> target is re-placed each morning until the position exits).

**Position limits** (configurable in `config/trading.json`):
- Max $2 per order
- Max $8 total position per ticker
- Fractional shares via Alpaca limit orders (qty-based)
- Stop-loss at `entry_price × stop_loss_pct` (default 0.95 = 5% below cost), enforced as an end-of-day market sell
- Max hold time of `max_hold_days` (default 15 trading-calendar days); positions that hit neither the profit target nor the stop are force-closed at market to free up capital for fresh setups

### Feedback loop

Every closed trade feeds back into the next night's run:
- Model scores update based on actual P&L percentage, not just win/loss labels. Real fills are weighted higher than theoretical OHLC checks, capped to avoid overreacting to one outlier, and decayed by 0.99× daily to emphasise recent accuracy.
- Each model's LLM prompt includes its own win rate and recent P&L, so it receives direct feedback on its performance.
- Over time, the consensus naturally shifts toward models that make accurate predictions.

---

## Quick start

```bash
pip install -r requirements.txt

# Pull Ollama models (config/models.json is the source of truth)
ollama pull qwen2.5:14b-instruct-q4_K_M
ollama pull deepseek-r1:8b

# Build ticker watchlist (top 200 liquid, low-volatility US equities,
# excluding leveraged/inverse/VIX-futures funds)
python3 update_tickers.py

# Run overnight analysis (or test with a small list)
python3 forge_loop.py --tickers NVDA,AAPL

# Dry-run the trading jobs (logs without placing or recording anything)
python3 trader.py --open --dry-run
python3 trader.py --close --dry-run
```

---

## Automation (GitHub Actions)

All workflows run on **self-hosted runners** (the morning, evening, and
report jobs inside Docker containers) and read Alpaca keys directly from
`~/.ssh/alpaca_paper_keys` (colon-delimited: `Key:`, `Secret_Key:`, `URL:`).
Keys are never stored in GitHub Secrets.

| Workflow | Schedule | What it does |
|---|---|---|
| [Nightly Forge](.github/workflows/nightly_forge.yml) | 23:00 UTC weekdays | Runs unit tests → `update_tickers.py` → `forge_loop.py` → validates outputs → refreshes backtest and walk-forward study → regenerates dashboard → commits state |
| [Morning Orders](.github/workflows/morning_orders.yml) | 13:30 UTC weekdays (9:30 AM EDT / 8:30 AM EST) | Places DAY limit buy orders for ACTIVE tickers; re-places the DAY profit-target sell for held positions |
| [Evening Cleanup](.github/workflows/evening_cleanup.yml) | 21:05 UTC weekdays (5:05 PM EDT / 4:05 PM EST) | Detects fills, records P&L, runs the end-of-day stop check, clears expired orders, refreshes trade attribution, regenerates dashboard |
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
| `python3 backtest.py` | Replay stored signals under the live trading rules |
| `python3 backtest.py --from-date 2026-05-01 --to-date 2026-05-15` | Backtest a bounded date window |
| `python3 backtest.py --stop-loss-pct 0.97 --max-hold-days 10` | What-if on execution parameters |
| `python3 backtest.py --walk-forward` | Also pick parameters on trailing windows and validate on the next |
| `python3 backtest.py --attribution` | Summarise actual closed-trade attribution by model, ticker, and execution quality |
| `python3 scripts/generate_html_report.py` | Regenerate `docs/index.html` from local data |

---

## Backtesting

`backtest.py` replays every stored signal day under the **live trading rules** and reports
what the strategy would have made, so its numbers are comparable with the live journal.

```bash
python3 backtest.py                                    # all history
python3 backtest.py --from-date 2026-07-01             # bounded window (inclusive)
python3 backtest.py --stop-loss-pct 0.97 --max-hold-days 10   # what-if on execution params
python3 backtest.py --walk-forward                     # plus the rolling train/test study
python3 backtest.py --attribution                      # break down actual closed trades
```

**What is simulated** (mirrors `trader.py`):
- **Signals as traded:** each `history/predictions_*.json` file is traded at the session its
  morning job used, with its stored ACTIVE classification. Nothing is recomputed from today's
  model scores, so there is no lookahead.
- **Entry:** DAY limit buy at `buy_high` at the open; fills at the open on a gap down, else at
  `buy_high` if the session trades down to it. One position per ticker, `max_per_trade_usd` each.
- **Target:** DAY limit sell at the signal's `sell_low`, working from the day **after** entry
  (live places it after the entry day's close); fills at the open on a gap up.
- **Stop and max hold:** checked against each close, entry day included. The market sell goes
  in after the close, so it fills at the **next open**. The report also scores the close that
  the live journal books (`strategy_as_booked`).
- **Prices:** Alpaca SIP daily bars (consolidated, unadjusted), downloaded once per run.

**Output** (`reports/backtest_summary.json`, rendered on the dashboard's **Backtest** tab, with
a dated snapshot in `reports/backtest_history/`):
- Strategy stats: trades, win rate, average return, profit factor, P&L in dollars, and a
  breakdown by exit (target / stop / max hold), with a significance read (`>= 30` trades).
- **Benchmark:** each trade against holding the same name over the same sessions.
- **By model:** each model's own ranges traded under the same rules.
- **Selection:** entry-session open-to-close return by stored signal (ACTIVE vs SKIP).
- **Calibration:** simulated trades matched to the live journal on (ticker, signal date), with
  win/loss agreement and the average return gap. Rows flagged `qty_desync_corrected` are left
  out, since their P&L reflects a share-count bug rather than the strategy.

**Walk-forward** (`reports/walk_forward_summary.json`): picks stop, max hold and a tighter
`min_upside_pct` on a trailing window of signal days, then validates on the next window.
Training is cut off before the test window starts, and windows without enough trades are
reported but left out of the averages. Evidence only: live config is never changed. The
nightly forge refreshes both reports.

**Trade attribution** (`reports/trade_attribution.json`): actual closed trades by model and
ticker, plus execution quality (entry improvement versus `buy_high`, target capture).

**Requirements:** at least one predictions file in `history/`, and the Alpaca keys (for the
bar download). The free data plan cannot query the last 15 minutes of SIP data, so today's
still-forming bar is always left out.

---

## Configuration

| File | Purpose | Key fields |
|---|---|---|
| `config/tickers.json` | Active watchlist (built by `update_tickers.py`) | — |
| `config/universe.json` | Ticker filter thresholds | `min_price`, `min_avg_daily_volume`, `max_daily_volatility_pct`, `max_tickers` |
| `config/trading.json` | Position limits and risk settings | `max_per_trade_usd` (2.0), `max_position_usd` (8.0), `stop_loss_pct` (0.95), `max_hold_days` (15), `score_decay_per_day` (0.99), `trade_score_scale` (0.02), `trade_score_cap` (0.10) |
| `config/signals.json` | Signal classification thresholds | `min_upside_pct` (1.5), `max_spread_pct` (3.0), `min_agreeing_models` (2), `max_consensus_cv` (0.10) |
| `config/models.json` | Model list (Ollama model IDs) | Never overwritten by automation |
| `state/analyst_scores.json` | Model scores — add/remove models here | 0.0–10.0 per model, initialised at 5.0 |

### config/trading.json

```json
{
    "max_per_trade_usd": 2.0,
    "max_position_usd": 8.0,
    "stop_loss_pct": 0.95,
    "max_hold_days": 15,
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
    "stop_order_id": null,
    "buy_limit": 890.50,
    "sell_limit": 910.00,
    "stop_limit": 846.00,
    "qty": 0.00225,
    "date": "2025-05-20",
    "pred_date": "2025-05-19",
    "closed": false
  }
}
```

- `sell_order_id` is the single resting profit-target order. The stop-loss is **not** a resting order (`stop_order_id` stays `null`); `stop_limit` records the stop *level* for reference, and the stop is enforced by the end-of-day check in `--close`.
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
python3 -m unittest discover -p 'test_*.py' -v
```

Discovery picks up every `test_*.py`; CI runs the same command before each nightly forge. Coverage:

| Module | Areas |
|---|---|
| `test_signals` | Consensus weighting, CV disagreement gate, fallback exclusion, signal classification, STALE upside handling |
| `test_backtest` | Win/stop/miss simulation, profit factor, avg win/loss %, max consecutive losses, internal field cleanup |
| `test_forge` | LLM output parsing (incl. inline reasoning blocks), fallback tagging, bar fetch window, prior-predictions lookup, RSI/Bollinger/%B/momentum technicals, score delta feedback, recency decay, stop-threshold evaluation |
| `test_trader` | `record_buy` / `record_sell`, P&L and qty-desync capping, partial vs full exits, EOD stop and max-hold, Alpaca read failures, market-session gate |
| `test_alpaca_client` | Order request construction, qty flooring, market calendar and last closed session, order lookup (404 vs error), position parsing |
| `test_update_tickers` | Leveraged/inverse/VIX fund exclusion |
| `test_no_banned_punctuation` | Site copy punctuation guard |

---

## License

MIT — see [LICENSE](LICENSE).
