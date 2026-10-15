#!/usr/bin/env python3
"""Generate a self-contained static HTML dashboard for GitHub Pages.

Reads:
  reports/signals_YYYY-MM-DD.json   (latest)
  history/trade_journal.json
  state/analyst_scores.json
  reports/backtest_summary.json     (optional)

Writes:
  docs/index.html
"""
from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from journal_quality import flagged_summary, split_journal  # noqa: E402

REPORTS_DIR = 'reports'
HISTORY_DIR = 'history'
SCORES_FILE = 'state/analyst_scores.json'
BACKTEST_FILE = 'reports/backtest_summary.json'
WALK_FORWARD_FILE = 'reports/walk_forward_summary.json'
ATTRIBUTION_FILE = 'reports/trade_attribution.json'
OUT_FILE = 'docs/index.html'

# GitHub repo, used by the in-page Rebuild button to trigger workflow_dispatch
GH_REPO = 'kdigitalsystems/OracleForge'
GH_WORKFLOW = 'regenerate_report.yml'
GH_BRANCH = 'main'


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_json(path: str, default):
    """Load JSON; a missing, empty or malformed file yields the default."""
    if not os.path.exists(path):
        return default
    try:
        with open(path, encoding='utf-8') as f:
            text = f.read()
        if not text.strip():
            print(f'WARNING: {path} is empty; using default.', file=sys.stderr)
            return default
        return json.loads(text)
    except (OSError, ValueError) as exc:
        print(f'WARNING: could not read {path} ({exc}); using default.', file=sys.stderr)
        return default


def find_latest_signals() -> tuple[str | None, dict | None]:
    """Return (date_str, report_dict) for the most recent signals file."""
    candidates = sorted(
        (n for n in os.listdir(REPORTS_DIR) if n.startswith('signals_') and n.endswith('.json')),
        reverse=True,
    ) if os.path.isdir(REPORTS_DIR) else []
    if not candidates:
        return None, None
    name = candidates[0]
    date_str = name.replace('signals_', '').replace('.json', '')
    return date_str, load_json(os.path.join(REPORTS_DIR, name), None)


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def _js_json(v) -> str:
    """json.dumps that is safe inside an inline <script> block."""
    return (json.dumps(v).replace('<', '\\u003c').replace('>', '\\u003e')
            .replace('&', '\\u0026'))


def _num(v, default: float = 0.0) -> float:
    """Float from anything; None, junk, NaN and inf give the default."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def _num_or_none(v):
    f = _num(v, float('nan'))
    return None if math.isnan(f) else f


def _str(v, default: str = '') -> str:
    return default if v is None else str(v)


def _dict(v) -> dict:
    return v if isinstance(v, dict) else {}


def _rows(v) -> list[dict]:
    return [r for r in v if isinstance(r, dict)] if isinstance(v, list) else []


def _money(v: float, digits: int = 2, signed: bool = False) -> str:
    """Dollar amount with the sign in front: -$196.84, never $-196.84."""
    v = _num(v)
    sign = '-' if v < 0 else ('+' if signed else '')
    return f'{sign}${abs(v):.{digits}f}'


def _esc(v) -> str:
    if v is None:
        return '-'
    s = str(v)
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#39;')


def _fmt_pct(v) -> str:
    if v is None:
        return '-'
    try:
        return f'{float(v):+.2f}%'
    except (ValueError, TypeError):
        return _esc(v)


def _fmt_usd(v) -> str:
    if v is None:
        return '-'
    try:
        return _money(float(v), 4)
    except (ValueError, TypeError):
        return _esc(v)


def _signal_badge(sig: str) -> str:
    colors = {
        'ACTIVE': 'bg-green-100 text-green-800',
        'STALE': 'bg-yellow-100 text-yellow-800',
        'SKIP': 'bg-gray-100 text-gray-600',
    }
    cls = colors.get(sig, 'bg-gray-100 text-gray-600')
    return f'<span class="px-2 py-0.5 rounded-full text-xs font-semibold {cls}">{_esc(sig)}</span>'


def _outcome_badge(outcome: str) -> str:
    cls = 'bg-green-100 text-green-800' if outcome == 'win' else 'bg-red-100 text-red-800'
    return f'<span class="px-2 py-0.5 rounded-full text-xs font-semibold {cls}">{_esc(outcome)}</span>'


def _table(headers: list[str], rows: list[list[str]], extra_cls: str = '') -> str:
    th = ''.join(
        f'<th scope="col" class="px-3 py-2 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">{h}</th>'
        for h in headers
    )
    body_rows = []
    for row in rows:
        tds = ''.join(f'<td class="px-3 py-2 text-sm text-gray-800 whitespace-nowrap">{cell}</td>' for cell in row)
        body_rows.append(f'<tr class="border-t border-gray-100 hover:bg-gray-50">{tds}</tr>')
    tbody = ''.join(body_rows)
    return f'''
<div class="overflow-x-auto {extra_cls}">
  <table class="min-w-full divide-y divide-gray-200">
    <thead class="bg-gray-50"><tr>{th}</tr></thead>
    <tbody class="bg-white">{tbody}</tbody>
  </table>
</div>'''


def _card(title: str, content: str, subtitle: str = '') -> str:
    sub = f'<p class="text-sm text-gray-500 mt-0.5">{subtitle}</p>' if subtitle else ''
    return f'''
<div class="bg-white rounded-xl shadow-sm border border-gray-200 p-5 mb-6">
  <h2 class="text-lg font-semibold text-gray-900">{title}</h2>
  {sub}
  <div class="mt-4">{content}</div>
</div>'''


def _metric(label: str, value: str, color: str = 'text-gray-900') -> str:
    return f'''
<div class="bg-gray-50 rounded-lg p-4 text-center">
  <div class="text-2xl font-bold {color}">{value}</div>
  <div class="text-xs text-gray-500 mt-1">{label}</div>
</div>'''


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _generated_text(raw) -> str:
    """Readable 'Generated ...' text with an explicit timezone note."""
    if raw is None:
        return 'Generated at unknown time'
    try:
        dt = datetime.fromisoformat(str(raw).strip().replace('Z', '+00:00'))
    except ValueError:
        return f'Generated {_esc(raw)}'
    if dt.tzinfo is None:
        return f'Generated {dt.strftime("%b %d, %Y %H:%M")} (runner clock, timezone not recorded)'
    return f'Generated {dt.astimezone(timezone.utc).strftime("%b %d, %Y %H:%M")} UTC'


def build_forecast_banner(date_str: str | None) -> str:
    """Amber banner when at least half the model predictions used the fallback range."""
    if not date_str:
        return ''
    path = os.path.join(HISTORY_DIR, f'predictions_{os.path.basename(str(date_str))}.json')
    if not os.path.exists(path):
        return ''
    preds = load_json(path, None)
    if not isinstance(preds, dict):
        return ''
    total = fallback = 0
    for entry in preds.values():
        models = entry.get('models') if isinstance(entry, dict) else None
        if not isinstance(models, dict):
            continue
        for m in models.values():
            # Only fallback predictions carry a 'fallback' key (set to True);
            # real ones have none, so count every non-skipped prediction.
            if not isinstance(m, dict) or m.get('skipped'):
                continue
            total += 1
            if m.get('fallback') is True:
                fallback += 1
    if total == 0 or fallback * 2 < total:
        return ''
    return (
        '<div role="alert" class="bg-amber-50 border border-amber-300 text-amber-900 rounded-lg px-4 py-3 mb-5 text-sm">'
        f'<span class="font-semibold">Forecast degraded: {fallback} of {total} model predictions used the '
        'fallback range, so few signals are shown.</span></div>'
    )


def build_signals_section(report: dict, date_str: str) -> str:
    summary = _dict(report.get('summary'))
    active_rows = _rows(report.get('active'))
    skip_rows = _rows(report.get('skip'))
    stale_rows = _rows(report.get('stale'))

    metrics = f'''
<div class="grid grid-cols-4 gap-3 mb-5">
  {_metric("Total", _esc(summary.get("total", 0)))}
  {_metric("ACTIVE", _esc(summary.get("active", 0)), "text-green-600")}
  {_metric("SKIP", _esc(summary.get("skip", 0)), "text-gray-500")}
  {_metric("STALE", _esc(summary.get("stale", 0)), "text-yellow-600")}
</div>'''

    # ACTIVE table
    if active_rows:
        hdrs = ['Ticker', 'Close', 'Buy Low', 'Buy High', 'Sell Low', 'Sell High', 'Upside %', 'Disagreement']
        tbl_rows = []
        for r in active_rows:
            upside = r.get('upside_pct')
            upside_str = f'<span class="font-semibold text-green-600">{_fmt_pct(upside)}</span>' if upside is not None else '-'
            cv = _num_or_none(r.get('consensus_cv'))
            # Coefficient of variation as a %; higher = models disagree more.
            if cv is None:
                cv_str = '-'
            else:
                cv_color = 'text-red-600' if cv > 0.05 else 'text-gray-500'
                cv_str = f'<span class="{cv_color}">{cv * 100:.1f}%</span>'
            tbl_rows.append([
                f'<span class="font-medium">{_esc(r.get("ticker"))}</span>',
                _esc(r.get('close')),
                _esc(r.get('buy_low')),
                _esc(r.get('buy_high')),
                _esc(r.get('sell_low')),
                _esc(r.get('sell_high')),
                upside_str,
                cv_str,
            ])
        active_section = f'<h3 class="font-semibold text-gray-700 mb-2">ACTIVE setups ({len(active_rows)})</h3>'
        active_section += _table(hdrs, tbl_rows)

        # Upside bar chart data
        chart_labels = _js_json([_str(r.get('ticker')) for r in active_rows[:20]])
        chart_data = _js_json([round(_num(r.get('upside_pct')), 2) for r in active_rows[:20]])
        active_section += f'''
<canvas id="upsideChart" class="mt-5" height="80"></canvas>
<script>
new Chart(document.getElementById("upsideChart"), {{
  type: "bar",
  data: {{
    labels: {chart_labels},
    datasets: [{{ label: "Upside %", data: {chart_data},
      backgroundColor: "rgba(34,197,94,0.7)", borderColor: "rgba(22,163,74,1)", borderWidth: 1 }}]
  }},
  options: {{ plugins: {{ legend: {{ display: false }} }},
    scales: {{ y: {{ beginAtZero: true, title: {{ display: true, text: "Upside %" }} }} }},
    responsive: true }}
}});
</script>'''
    else:
        active_section = '<p class="text-gray-500 text-sm">No ACTIVE setups for this date.</p>'

    # Collapsed SKIP + STALE
    all_other = stale_rows + skip_rows
    other_html = ''
    if all_other:
        hdrs2 = ['Ticker', 'Signal', 'Close', 'Buy High', 'Sell Low', 'Upside %']
        other_rows = []
        for r in all_other:
            other_rows.append([
                f'<span class="font-medium">{_esc(r.get("ticker"))}</span>',
                _signal_badge(_str(r.get('signal'))),
                _esc(r.get('close')),
                _esc(r.get('buy_high')),
                _esc(r.get('sell_low')),
                _fmt_pct(r.get('upside_pct')),
            ])
        other_html = f'''
<details class="mt-4">
  <summary class="cursor-pointer text-sm text-gray-500 hover:text-gray-700">
    Show all other tickers ({len(all_other)})
  </summary>
  <div class="mt-2">{_table(hdrs2, other_rows)}</div>
</details>'''

    content = metrics + active_section + other_html
    return _card(f'Signals: {date_str}', content,
                 subtitle=_generated_text(report.get('generated_at')))


def _bar_pct(score) -> float:
    return max(0.0, min(100.0, _num(score) / 10.0 * 100))


def build_scores_section(scores: dict) -> str:
    if not scores:
        return _card('Model Scores', '<p class="text-gray-500 text-sm">No scores found.</p>')

    sorted_scores = sorted(((m, _num(v)) for m, v in scores.items()), key=lambda x: x[1], reverse=True)

    bars_html = ''
    for model, score in sorted_scores:
        pct = _bar_pct(score)
        color = 'bg-green-500' if score >= 7 else 'bg-yellow-400' if score >= 4 else 'bg-red-400'
        bars_html += f'''
<div class="flex items-center gap-3 mb-2">
  <span class="text-sm text-gray-700 w-56 truncate" title="{_esc(model)}">{_esc(model)}</span>
  <div class="flex-1 bg-gray-100 rounded-full h-4">
    <div class="{color} h-4 rounded-full transition-all" style="width:{pct:.1f}%"></div>
  </div>
  <span class="text-sm font-semibold text-gray-800 w-10 text-right">{score:.1f}</span>
</div>'''

    return _card('Model Scores (MoE weights)', bars_html,
                 subtitle='Scores range 0 to 10. Updated nightly by OHLC feedback loop.')


def _pnl_note(flagged: list[dict]) -> str:
    if not flagged:
        return ''
    info = flagged_summary(flagged)
    count = int(info.get('count') or 0)
    if count <= 0:
        return ''
    names = {
        'qty_desync': 'share-count desync',
        'near_zero_invested': 'near-zero cost basis',
        'sign_mismatch': 'sign mismatch',
        'unparseable': 'unreadable fields',
    }
    reasons = ', '.join(names.get(k, _esc(k).replace('_', ' ')) for k in _dict(info.get('reasons')))
    reasons = f' ({reasons})' if reasons else ''
    noun, verb = ('trade', 'is') if count == 1 else ('trades', 'are')
    return (f'<p class="text-sm text-gray-500 mb-5">{count} earlier {noun} with unreliable accounting'
            f'{reasons} {verb} left out. As recorded they sum to '
            f'{_esc(_money(info.get("recorded_pnl_usd"), 2))}.</p>')


def build_pnl_section(journal: list[dict]) -> str:
    journal = _rows(journal)
    reliable, flagged = split_journal(journal)
    note = _pnl_note(flagged)
    if not reliable:
        msg = 'No trades with reliable accounting yet.' if flagged else 'No closed trades yet.'
        return _card('P&amp;L Trade Journal', f'<p class="text-gray-500 text-sm">{msg}</p>{note}')

    n = len(reliable)
    total_pnl = sum(_num(t.get('pnl_usd')) for t in reliable)
    wins = sum(1 for t in reliable if t.get('outcome') == 'win')
    win_rate = wins / n * 100
    avg_pct = sum(_num(t.get('pnl_pct')) for t in reliable) / n
    best = max(reliable, key=lambda t: _num(t.get('pnl_usd')))
    worst = min(reliable, key=lambda t: _num(t.get('pnl_usd')))

    pnl_color = 'text-green-600' if total_pnl >= 0 else 'text-red-600'
    avg_color = 'text-green-600' if avg_pct >= 0 else 'text-red-600'

    def _trade_metric(label: str, t: dict) -> str:
        v = _num(t.get('pnl_usd'))
        return _metric(f'{label} ({_esc(t.get("ticker"))})', _esc(_money(v, 2, True)),
                       'text-green-600' if v >= 0 else 'text-red-600')

    metrics = f'''
<div class="grid grid-cols-2 md:grid-cols-6 gap-3 mb-5">
  {_metric("Total trades", str(n))}
  {_metric("Win rate", f"{win_rate:.0f}%", "text-green-600" if win_rate >= 50 else "text-red-600")}
  {_metric("Total P&amp;L", _esc(_money(total_pnl, 4, True)), pnl_color)}
  {_metric("Avg P&amp;L %", f"{avg_pct:+.2f}%", avg_color)}
  {_trade_metric("Best trade", best)}
  {_trade_metric("Worst trade", worst)}
</div>{note}'''

    # Cumulative P&L chart
    by_date: dict[str, float] = {}
    for t in sorted(reliable, key=lambda x: _str(x.get('close_date'), 'unknown')):
        d = _str(t.get('close_date'), 'unknown')
        by_date[d] = by_date.get(d, 0) + _num(t.get('pnl_usd'))
    dates_sorted = sorted(by_date)
    cum_pnl = []
    running = 0.0
    for d in dates_sorted:
        running += by_date[d]
        cum_pnl.append(round(running, 4))

    chart_html = ''
    if len(dates_sorted) > 1:
        chart_html = f'''
<canvas id="pnlChart" height="80" class="mb-5"></canvas>
<script>
new Chart(document.getElementById("pnlChart"), {{
  type: "line",
  data: {{
    labels: {_js_json(dates_sorted)},
    datasets: [{{ label: "Cumulative P&L ($)", data: {_js_json(cum_pnl)},
      borderColor: "rgba(59,130,246,1)", backgroundColor: "rgba(59,130,246,0.1)",
      fill: true, tension: 0.3 }}]
  }},
  options: {{ plugins: {{ legend: {{ display: false }} }},
    scales: {{ y: {{ title: {{ display: true, text: "P&L ($)" }} }} }},
    responsive: true }}
}});
</script>'''

    # Recent trades table
    recent = sorted(reliable, key=lambda x: _str(x.get('close_date')), reverse=True)[:20]
    hdrs = ['Date', 'Ticker', 'Entry', 'Exit', 'P&L $', 'P&L %', 'Outcome']
    tbl_rows = []
    for t in recent:
        pnl_usd = _num(t.get('pnl_usd'))
        pnl_pct = _num(t.get('pnl_pct'))
        pnl_usd_str = f'<span class="{"text-green-600" if pnl_usd >= 0 else "text-red-600"} font-medium">{_fmt_usd(pnl_usd)}</span>'
        pnl_pct_str = f'<span class="{"text-green-600" if pnl_pct >= 0 else "text-red-600"}">{_fmt_pct(pnl_pct)}</span>'
        tbl_rows.append([
            _esc(t.get('close_date')),
            f'<span class="font-medium">{_esc(t.get("ticker"))}</span>',
            _esc(t.get('entry_price')),
            _esc(t.get('exit_price')),
            pnl_usd_str,
            pnl_pct_str,
            _outcome_badge(_str(t.get('outcome'))),
        ])
    trades_table = '<h3 class="font-semibold text-gray-700 mb-2">Recent trades</h3>' + _table(hdrs, tbl_rows)

    content = metrics + chart_html + trades_table
    return _card('P&amp;L Trade Journal', content)


def build_backtest_section(report: dict) -> str:
    if not report or 'strategy' not in report:
        return ''
    strat = _dict(report['strategy'])
    params = _dict(report.get('params'))
    edge = _dict(_dict(report.get('benchmark')).get('edge_vs_buy_hold'))
    calib = _dict(report.get('calibration'))

    def _pf(v) -> str:
        v = _num_or_none(v)
        if v is None:
            return 'N/A'
        color = 'text-green-600' if v >= 1.0 else 'text-red-600'
        return f'<span class="{color} font-medium">{v:.2f}</span>'

    def _ret(v) -> str:
        v = _num(v)
        color = 'text-green-600' if v > 0 else 'text-red-600'
        return f'<span class="{color} font-medium">{v:+.2f}%</span>'

    edge_txt = _fmt_pct(edge.get('mean')) if edge.get('n') else '-'
    if edge.get('n'):
        edge_note = 'significant' if edge.get('significant') else 'not significant'
        edge_txt += f'<div class="text-xs text-gray-500 font-normal">{edge_note}, n={_esc(edge.get("n"))}</div>'
    metrics = f'''
<div class="grid grid-cols-2 md:grid-cols-5 gap-3 mb-5">
  {_metric("Closed Trades", _esc(strat.get("trades", 0)))}
  {_metric("Win Rate", f'{_num(strat.get("win_rate")) * 100:.1f}%')}
  {_metric("Avg Return", _fmt_pct(strat.get("avg_return_pct")))}
  {_metric("Profit Factor", "N/A" if _num_or_none(strat.get("profit_factor")) is None else f'{_num(strat.get("profit_factor")):.2f}')}
  {_metric("Edge vs Hold", edge_txt)}
</div>'''

    reason_rows = [
        [_esc(reason), _esc(r.get('trades', 0)), _ret(r.get('avg_return_pct')),
         _esc(_money(r.get('total_pnl_usd'), 2, True))]
        for reason, r in _dict(strat.get('by_exit_reason')).items() if isinstance(r, dict)
    ]
    model_rows = [
        [_esc(model), _esc(m.get('trades', 0)), f'{_num(m.get("win_rate")) * 100:.1f}%',
         _ret(m.get('avg_return_pct')), _pf(m.get('profit_factor')), _esc(m.get('max_consecutive_losses', 0))]
        for model, m in sorted(((k, v) for k, v in _dict(report.get('by_model')).items() if isinstance(v, dict)),
                               key=lambda x: _num(x[1].get('avg_return_pct')), reverse=True)
    ]
    content = metrics
    if reason_rows:
        content += '<h3 class="font-semibold text-gray-700 mb-2">By exit</h3>'
        content += _table(['Exit', 'Trades', 'Avg Return', 'P&amp;L'], reason_rows)
    if model_rows:
        content += '<h3 class="font-semibold text-gray-700 mt-5 mb-2">Each model on its own ranges</h3>'
        content += _table(['Model', 'Trades', 'Win %', 'Avg Return', 'Profit Factor', 'Max Consec. Losses'], model_rows)
    if calib.get('matched'):
        content += (
            '<p class="text-sm text-gray-500 mt-4">Calibration against the live journal: '
            f'{_esc(calib["matched"])} matched trades agree on win or loss '
            f'{_num(calib.get("agreement_rate")) * 100:.0f}% of the time '
            f'(live avg {_num(calib.get("avg_live_return_pct")):+.2f}%, '
            f'simulated {_num(calib.get("avg_sim_booked_return_pct")):+.2f}%).</p>'
        )

    days = _esc(report.get('days_in_history', 0))
    subtitle = (f'Replay of stored ACTIVE signals under the live trading rules: {days} signal day(s), '
                f'${_num(params.get("order_usd")):.2f} per order')
    stop = _num_or_none(params.get('stop_loss_pct'))
    if stop is not None:
        subtitle += (f', stop {(1 - stop) * 100:.0f}% below entry, '
                     f'max hold {_esc(params.get("max_hold_days"))}d')
    return _card('Backtest', content, subtitle=subtitle)


def build_walk_forward_section(report: dict) -> str:
    if not report or 'summary' not in report:
        return ''
    summary = _dict(report['summary'])
    rec = _dict(report.get('latest_recommendation'))
    params = _dict(rec.get('selected_params'))
    metrics = f'''
<div class="grid grid-cols-2 md:grid-cols-4 gap-3 mb-5">
  {_metric("Windows", f'{_esc(summary.get("validation_windows", 0))} of {_esc(summary.get("windows", 0))}')}
  {_metric("Validation Trades", _esc(summary.get("validation_trades", 0)))}
  {_metric("Avg Validation Return", _fmt_pct(summary.get("avg_validation_return_pct")))}
  {_metric("Positive Windows", _esc(summary.get("positive_windows", 0)))}
</div>'''
    content = metrics
    if params:
        stats = _dict(rec.get('train_stats'))
        train = rec.get('train')
        span = (f'{_esc(train[0])} to {_esc(train[1])}, '
                if isinstance(train, (list, tuple)) and len(train) >= 2 else '')
        content += (f'<h3 class="font-semibold text-gray-700 mb-2">Latest pick '
                    f'({span}n={_esc(stats.get("n", 0))})</h3>')
        content += _table(['Parameter', 'Value'], [[_esc(k), _esc(v)] for k, v in params.items()])
    elif rec:
        content += ('<p class="text-sm text-gray-500">Too few trades in the latest window to pick parameters '
                    f'(need {_esc(report.get("min_train_trades"))}).</p>')
    return _card(
        'Walk-Forward Study',
        content,
        subtitle=('Parameters picked on a trailing window and validated on the next one; '
                  'windows without trades are left out. Live config is never changed.'),
    )


def build_attribution_section(report: dict) -> str:
    if not report:
        return ''
    overall = _dict(report.get('overall'))
    eq = _dict(report.get('execution_quality'))
    metrics = f'''
<div class="grid grid-cols-4 gap-3 mb-5">
  {_metric("Trades", _esc(report.get("trades", 0)))}
  {_metric("Win Rate", f'{_num(overall.get("win_rate")) * 100:.1f}%')}
  {_metric("Avg P&L", _fmt_pct(overall.get("avg_pnl_pct")))}
  {_metric("Entry Improvement", _fmt_pct(eq.get("avg_price_improvement_pct")))}
</div>'''
    rows = []
    for model, stats in [(k, v) for k, v in _dict(report.get('by_model')).items() if isinstance(v, dict)][:8]:
        rows.append([
            _esc(model),
            _esc(stats.get('trades', 0)),
            f'{_num(stats.get("win_rate")) * 100:.1f}%',
            _fmt_pct(stats.get('avg_pnl_pct')),
            _fmt_pct(stats.get('total_pnl_pct')),
        ])
    content = metrics
    if rows:
        content += '<h3 class="font-semibold text-gray-700 mb-2">Model attribution</h3>'
        content += _table(['Model', 'Trades', 'Win %', 'Avg P&L', 'Total P&L'], rows)
    subtitle = ''
    excluded_n = int(_num(_dict(report.get('excluded')).get('count')))
    if excluded_n > 0:
        noun = 'trade' if excluded_n == 1 else 'trades'
        subtitle = f'{excluded_n} {noun} with unreliable accounting excluded'
    return _card('Trade Attribution', content, subtitle=subtitle)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate() -> None:
    os.makedirs('docs', exist_ok=True)

    date_str, report = find_latest_signals()
    journal = load_json('history/trade_journal.json', [])
    scores = load_json(SCORES_FILE, {})
    backtest = load_json(BACKTEST_FILE, None)
    walk_forward = load_json(WALK_FORWARD_FILE, None)
    attribution = load_json(ATTRIBUTION_FILE, None)
    report = report if isinstance(report, dict) else None
    journal = journal if isinstance(journal, list) else []
    scores = scores if isinstance(scores, dict) else {}
    backtest = backtest if isinstance(backtest, dict) else None
    walk_forward = walk_forward if isinstance(walk_forward, dict) else None
    attribution = attribution if isinstance(attribution, dict) else None

    if not report and not journal and not scores:
        print('WARNING: No data found. Generating empty placeholder page.')

    # Embed build time as both a display string and an ISO timestamp for JS
    now = datetime.now(timezone.utc)
    generated_display = now.strftime('%b %d, %Y  %H:%M UTC')
    generated_iso = now.strftime('%Y-%m-%dT%H:%M:%SZ')

    def _safe(fn, *args) -> str:
        # One bad section must not abort the whole page (the atomic write
        # would then leave a stale page behind while the nightly fails).
        try:
            return fn(*args)
        except Exception as exc:  # noqa: BLE001
            print(f'WARNING: {fn.__name__} failed: {exc!r}', file=sys.stderr)
            return _card('Section unavailable', '<p class="text-gray-500 text-sm">This section could not be rendered.</p>')

    signals_html = (
        _safe(build_forecast_banner, date_str) + _safe(build_signals_section, report, date_str)
    ) if report else _card(
        'Signals', '<p class="text-gray-500 text-sm">No signals report found. Run forge_loop.py first.</p>'
    )
    scores_html = _safe(build_scores_section, scores)
    pnl_html = _safe(build_pnl_section, journal)
    backtest_html = (
        _safe(build_backtest_section, backtest)
        + _safe(build_walk_forward_section, walk_forward)
        + _safe(build_attribution_section, attribution)
    )

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>OracleForge</title>
  <link rel="icon" type="image/svg+xml" href="favicon.svg" />
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
    details summary {{ list-style: none; }}
    details summary::before {{ content: "▶ "; }}
    details[open] summary::before {{ content: "▼ "; }}
    #rebuild-btn:disabled {{ opacity: 0.6; cursor: not-allowed; }}
  </style>
</head>
<body class="bg-gray-50 min-h-screen">

  <!-- Token modal -->
  <div id="token-modal" role="dialog" aria-modal="true" aria-labelledby="token-modal-title" class="hidden fixed inset-0 bg-black bg-opacity-40 z-50 flex items-center justify-center">
    <div class="bg-white rounded-xl shadow-xl p-6 w-full max-w-md mx-4">
      <h3 id="token-modal-title" class="text-lg font-semibold text-gray-900 mb-1">GitHub Personal Access Token</h3>
      <p class="text-sm text-gray-500 mb-3">
        Required once to trigger workflow rebuilds. Stored only in your browser's localStorage;
        never sent anywhere except GitHub's API.<br><br>
        Create one at <a href="https://github.com/settings/tokens/new?scopes=workflow&description=OracleForge+rebuild"
          target="_blank" class="text-blue-600 underline">github.com → Settings → Tokens</a>
        with the <code class="bg-gray-100 px-1 rounded">workflow</code> scope.
      </p>
      <input id="token-input" type="password" placeholder="ghp_xxxxxxxxxxxxxxxxxxxx"
        class="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm font-mono mb-4 focus:outline-none focus:ring-2 focus:ring-blue-400" />
      <div class="flex gap-2 justify-end">
        <button onclick="closeModal()" class="px-4 py-2 text-sm text-gray-600 hover:text-gray-800">Cancel</button>
        <button onclick="saveToken()"
          class="px-4 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 font-medium">Save &amp; Rebuild</button>
      </div>
    </div>
  </div>

  <!-- Rebuild status bar (hidden until triggered) -->
  <div id="rebuild-bar" class="hidden fixed bottom-0 left-0 right-0 bg-blue-600 text-white text-sm py-2 px-4 flex items-center justify-between z-40">
    <span id="rebuild-msg">⏳ Rebuilding report…</span>
    <span id="rebuild-countdown" class="font-mono text-blue-200"></span>
  </div>

  <!-- Header -->
  <header class="bg-white border-b border-gray-200 shadow-sm sticky top-0 z-10">
    <div class="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between gap-4">
      <div>
        <h1 class="text-xl font-bold text-gray-900">⚡ OracleForge</h1>
        <p class="text-xs text-gray-500">LLM ensemble · Alpaca paper trading · Auto-updated nightly</p>
      </div>

      <!-- Build time + Rebuild button -->
      <div class="flex items-center gap-3 shrink-0">
        <div class="text-right">
          <div class="text-xs font-medium text-gray-700">Last built</div>
          <div class="text-xs text-gray-400" id="build-time-rel" title="{generated_display}">{generated_display}</div>
        </div>
        <button id="rebuild-btn" onclick="onRebuildClick()"
          class="flex items-center gap-1.5 px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold rounded-lg transition-colors">
          <svg xmlns="http://www.w3.org/2000/svg" class="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.5">
            <path stroke-linecap="round" stroke-linejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          Rebuild
        </button>
        <button onclick="clearToken()" title="Clear saved GitHub token" aria-label="Clear saved GitHub token"
          class="text-gray-600 hover:text-gray-900 transition-colors text-lg leading-none">⚙</button>
      </div>
    </div>
  </header>

  <!-- Nav tabs -->
  <div class="max-w-6xl mx-auto px-4 mt-4">
    <div role="tablist" class="flex gap-2 mb-5 border-b border-gray-200">
      <button onclick="showTab('signals')" id="tab-signals" role="tab" aria-selected="true" aria-controls="pane-signals"
        class="tab-btn px-4 py-2 text-sm font-medium text-blue-600 border-b-2 border-blue-600">Signals</button>
      <button onclick="showTab('pnl')" id="tab-pnl" role="tab" aria-selected="false" aria-controls="pane-pnl"
        class="tab-btn px-4 py-2 text-sm font-medium text-gray-500 hover:text-gray-700 border-b-2 border-transparent">P&amp;L</button>
      <button onclick="showTab('backtest')" id="tab-backtest" role="tab" aria-selected="false" aria-controls="pane-backtest"
        class="tab-btn px-4 py-2 text-sm font-medium text-gray-500 hover:text-gray-700 border-b-2 border-transparent">Backtest</button>
      <button onclick="showTab('models')" id="tab-models" role="tab" aria-selected="false" aria-controls="pane-models"
        class="tab-btn px-4 py-2 text-sm font-medium text-gray-500 hover:text-gray-700 border-b-2 border-transparent">Models</button>
    </div>

    <div id="pane-signals" role="tabpanel" aria-labelledby="tab-signals">{signals_html}</div>
    <div id="pane-pnl" role="tabpanel" aria-labelledby="tab-pnl" class="hidden">{pnl_html}</div>
    <div id="pane-backtest" role="tabpanel" aria-labelledby="tab-backtest" class="hidden">{backtest_html if backtest_html else _card("Backtest", '<p class="text-gray-500 text-sm">No backtest report found. Run python backtest.py to generate one.</p>')}</div>
    <div id="pane-models" role="tabpanel" aria-labelledby="tab-models" class="hidden">{scores_html}</div>
  </div>

  <script>
    // ── Tab switching ──────────────────────────────────────────────────────
    function showTab(name) {{
      ['signals','pnl','backtest','models'].forEach(t => {{
        document.getElementById('pane-' + t).classList.toggle('hidden', t !== name);
        const btn = document.getElementById('tab-' + t);
        btn.setAttribute('aria-selected', t === name ? 'true' : 'false');
        if (t === name) {{
          btn.classList.replace('text-gray-500','text-blue-600');
          btn.classList.replace('border-transparent','border-blue-600');
        }} else {{
          btn.classList.replace('text-blue-600','text-gray-500');
          btn.classList.replace('border-blue-600','border-transparent');
        }}
      }});
    }}

    // ── Relative build time ────────────────────────────────────────────────
    (function() {{
      const built = new Date('{generated_iso}');
      function relTime() {{
        const diff = Math.floor((Date.now() - built) / 1000);
        if (diff < 60)  return diff + 's ago';
        if (diff < 3600) return Math.floor(diff/60) + 'm ago';
        if (diff < 86400) return Math.floor(diff/3600) + 'h ago';
        return Math.floor(diff/86400) + 'd ago';
      }}
      const el = document.getElementById('build-time-rel');
      if (el) {{ el.textContent = relTime(); setInterval(() => el.textContent = relTime(), 30000); }}
    }})();

    // ── GitHub token helpers ───────────────────────────────────────────────
    const REPO     = '{GH_REPO}';
    const WORKFLOW = '{GH_WORKFLOW}';
    const BRANCH   = '{GH_BRANCH}';

    function getToken()  {{ return localStorage.getItem('gh_token'); }}
    function clearToken() {{
      localStorage.removeItem('gh_token');
      alert('GitHub token cleared. You will be prompted again on the next rebuild.');
    }}
    function closeModal() {{ document.getElementById('token-modal').classList.add('hidden'); }}
    document.addEventListener('keydown', e => {{ if (e.key === 'Escape') closeModal(); }});
    function saveToken() {{
      const t = document.getElementById('token-input').value.trim();
      if (!t) {{ alert('Please enter a token.'); return; }}
      localStorage.setItem('gh_token', t);
      closeModal();
      triggerRebuild(t);
    }}

    // ── Rebuild flow ───────────────────────────────────────────────────────
    function onRebuildClick() {{
      const token = getToken();
      if (!token) {{
        document.getElementById('token-modal').classList.remove('hidden');
        document.getElementById('token-input').focus();
      }} else {{
        triggerRebuild(token);
      }}
    }}

    async function triggerRebuild(token) {{
      const btn = document.getElementById('rebuild-btn');
      btn.disabled = true;

      try {{
        const res = await fetch(
          `https://api.github.com/repos/${{REPO}}/actions/workflows/${{WORKFLOW}}/dispatches`,
          {{
            method: 'POST',
            headers: {{
              'Authorization': `Bearer ${{token}}`,
              'Accept': 'application/vnd.github+json',
              'Content-Type': 'application/json',
            }},
            body: JSON.stringify({{ ref: BRANCH }}),
          }}
        );

        if (res.status === 401 || res.status === 403) {{
          localStorage.removeItem('gh_token');
          alert('GitHub token rejected (status ' + res.status + '). Please re-enter it.');
          btn.disabled = false;
          return;
        }}
        if (!res.ok) {{
          const body = await res.text();
          alert('GitHub API error ' + res.status + ':\\n' + body);
          btn.disabled = false;
          return;
        }}
      }} catch (err) {{
        alert('Network error: ' + err.message);
        btn.disabled = false;
        return;
      }}

      // Show countdown bar and reload when it hits 0
      startCountdown(40);
    }}

    function startCountdown(seconds) {{
      const bar = document.getElementById('rebuild-bar');
      const msg = document.getElementById('rebuild-msg');
      const cntEl = document.getElementById('rebuild-countdown');
      bar.classList.remove('hidden');

      let remaining = seconds;
      cntEl.textContent = 'Reloading in ' + remaining + 's…';

      const iv = setInterval(() => {{
        remaining--;
        if (remaining <= 0) {{
          clearInterval(iv);
          msg.textContent = '✅ Done! Reloading…';
          cntEl.textContent = '';
          // Hard reload: bypass cache so we get the newly committed HTML
          location.href = location.href.split('?')[0] + '?v=' + Date.now();
        }} else {{
          cntEl.textContent = 'Reloading in ' + remaining + 's…';
          if (remaining <= 10) msg.textContent = '⏳ Almost there…';
        }}
      }}, 1000);
    }}
  </script>

</body>
</html>'''

    # Write-then-rename: same reasoning as the atomic save_json() in
    # trader.py/forge_loop.py/signals.py/update_tickers.py -- avoids a
    # concurrent reader (or a second workflow's checkout on the same
    # self-hosted runner) ever seeing a partially-written file.
    tmp_path = f'{OUT_FILE}.tmp{os.getpid()}'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        f.write(html)
    os.replace(tmp_path, OUT_FILE)

    print(f'Generated {OUT_FILE} ({os.path.getsize(OUT_FILE):,} bytes)')


if __name__ == '__main__':
    generate()
