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
import os
import sys
from datetime import datetime, timezone

REPORTS_DIR = 'reports'
HISTORY_DIR = 'history'
SCORES_FILE = 'state/analyst_scores.json'
BACKTEST_FILE = 'reports/backtest_summary.json'
OUT_FILE = 'docs/index.html'

# GitHub repo — used by the in-page Rebuild button to trigger workflow_dispatch
GH_REPO = 'kdigitalsystems/OracleForge'
GH_WORKFLOW = 'regenerate_report.yml'
GH_BRANCH = 'main'


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_json(path: str, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
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

def _esc(v) -> str:
    if v is None:
        return '—'
    s = str(v)
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#39;')


def _fmt_pct(v) -> str:
    if v is None:
        return '—'
    try:
        return f'{float(v):+.2f}%'
    except (ValueError, TypeError):
        return str(v)


def _fmt_usd(v) -> str:
    if v is None:
        return '—'
    try:
        return f'${float(v):.4f}'
    except (ValueError, TypeError):
        return str(v)


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
        f'<th class="px-3 py-2 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">{h}</th>'
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

def build_signals_section(report: dict, date_str: str) -> str:
    summary = report.get('summary', {})
    active_rows = report.get('active', [])
    skip_rows = report.get('skip', [])
    stale_rows = report.get('stale', [])

    metrics = f'''
<div class="grid grid-cols-4 gap-3 mb-5">
  {_metric("Total", str(summary.get("total", 0)))}
  {_metric("ACTIVE", str(summary.get("active", 0)), "text-green-600")}
  {_metric("SKIP", str(summary.get("skip", 0)), "text-gray-500")}
  {_metric("STALE", str(summary.get("stale", 0)), "text-yellow-600")}
</div>'''

    # ACTIVE table
    if active_rows:
        hdrs = ['Ticker', 'Close', 'Buy Low', 'Buy High', 'Sell Low', 'Sell High', 'Upside %']
        tbl_rows = []
        for r in active_rows:
            upside = r.get('upside_pct')
            upside_str = f'<span class="font-semibold text-green-600">{_fmt_pct(upside)}</span>' if upside else '—'
            tbl_rows.append([
                f'<span class="font-medium">{_esc(r.get("ticker"))}</span>',
                _esc(r.get('close')),
                _esc(r.get('buy_low')),
                _esc(r.get('buy_high')),
                _esc(r.get('sell_low')),
                _esc(r.get('sell_high')),
                upside_str,
            ])
        active_section = f'<h3 class="font-semibold text-gray-700 mb-2">ACTIVE setups ({len(active_rows)})</h3>'
        active_section += _table(hdrs, tbl_rows)

        # Upside bar chart data
        chart_labels = json.dumps([r.get('ticker', '') for r in active_rows[:20]])
        chart_data = json.dumps([round(float(r.get('upside_pct') or 0), 2) for r in active_rows[:20]])
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
                _signal_badge(r.get('signal', '')),
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
    return _card(f'Signals — {date_str}', content,
                 subtitle=f'Generated at {report.get("generated_at", "—")}')


def build_scores_section(scores: dict) -> str:
    if not scores:
        return _card('Model Scores', '<p class="text-gray-500 text-sm">No scores found.</p>')

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    max_score = max(v for _, v in sorted_scores) or 10.0

    bars_html = ''
    for model, score in sorted_scores:
        pct = (score / 10.0) * 100
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
                 subtitle='Scores range 0–10. Updated nightly by OHLC feedback loop.')


def build_pnl_section(journal: list[dict]) -> str:
    if not journal:
        return _card('P&amp;L — Trade Journal',
                     '<p class="text-gray-500 text-sm">No closed trades yet.</p>')

    total_pnl = sum(t.get('pnl_usd', 0) for t in journal)
    wins = sum(1 for t in journal if t.get('outcome') == 'win')
    losses = len(journal) - wins
    win_rate = (wins / len(journal) * 100) if journal else 0
    avg_pct = sum(t.get('pnl_pct', 0) for t in journal) / len(journal)

    pnl_color = 'text-green-600' if total_pnl >= 0 else 'text-red-600'
    avg_color = 'text-green-600' if avg_pct >= 0 else 'text-red-600'

    metrics = f'''
<div class="grid grid-cols-4 gap-3 mb-5">
  {_metric("Total trades", str(len(journal)))}
  {_metric("Win rate", f"{win_rate:.0f}%", "text-green-600" if win_rate >= 50 else "text-red-600")}
  {_metric("Total P&amp;L", f"${total_pnl:+.4f}", pnl_color)}
  {_metric("Avg P&amp;L %", f"{avg_pct:+.2f}%", avg_color)}
</div>'''

    # Cumulative P&L chart
    by_date: dict[str, float] = {}
    for t in sorted(journal, key=lambda x: x.get('close_date', '')):
        d = t.get('close_date', 'unknown')
        by_date[d] = by_date.get(d, 0) + t.get('pnl_usd', 0)
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
    labels: {json.dumps(dates_sorted)},
    datasets: [{{ label: "Cumulative P&L ($)", data: {json.dumps(cum_pnl)},
      borderColor: "rgba(59,130,246,1)", backgroundColor: "rgba(59,130,246,0.1)",
      fill: true, tension: 0.3 }}]
  }},
  options: {{ plugins: {{ legend: {{ display: false }} }},
    scales: {{ y: {{ title: {{ display: true, text: "P&L ($)" }} }} }},
    responsive: true }}
}});
</script>'''

    # Recent trades table
    recent = sorted(journal, key=lambda x: x.get('close_date', ''), reverse=True)[:20]
    hdrs = ['Date', 'Ticker', 'Entry', 'Exit', 'P&L $', 'P&L %', 'Outcome']
    tbl_rows = []
    for t in recent:
        pnl_usd = t.get('pnl_usd', 0)
        pnl_pct = t.get('pnl_pct', 0)
        pnl_usd_str = f'<span class="{"text-green-600" if pnl_usd >= 0 else "text-red-600"} font-medium">{_fmt_usd(pnl_usd)}</span>'
        pnl_pct_str = f'<span class="{"text-green-600" if pnl_pct >= 0 else "text-red-600"}">{_fmt_pct(pnl_pct)}</span>'
        tbl_rows.append([
            _esc(t.get('close_date')),
            f'<span class="font-medium">{_esc(t.get("ticker"))}</span>',
            _esc(t.get('entry_price')),
            _esc(t.get('exit_price')),
            pnl_usd_str,
            pnl_pct_str,
            _outcome_badge(t.get('outcome', '')),
        ])
    trades_table = f'<h3 class="font-semibold text-gray-700 mb-2">Recent trades</h3>' + _table(hdrs, tbl_rows)

    content = metrics + chart_html + trades_table
    return _card('P&amp;L — Trade Journal', content)


def build_backtest_section(report: dict) -> str:
    if not report:
        return ''

    by_signal = report.get('by_signal', {})
    by_model = report.get('by_model', {})

    signal_hdrs = ['Signal', 'Trades', 'Triggered', 'Win %', 'Avg Return %']
    signal_rows = []
    for sig, s in sorted(by_signal.items()):
        triggered = s.get('triggered', 0)
        win_pct = f'{s.get("win_rate", 0) * 100:.1f}%'
        avg_ret = f'{s.get("avg_return_pct", 0):.2f}%'
        color = 'text-green-600' if s.get('avg_return_pct', 0) > 0 else 'text-red-600'
        signal_rows.append([
            _signal_badge(sig) if sig in ('ACTIVE', 'SKIP', 'STALE') else _esc(sig),
            str(s.get('trades', 0)),
            str(triggered),
            win_pct,
            f'<span class="{color} font-medium">{avg_ret}</span>',
        ])

    model_hdrs = ['Model', 'Trades', 'Triggered', 'Win %', 'Avg Return %']
    model_rows = []
    for model, s in sorted(by_model.items(), key=lambda x: x[1].get('win_rate', 0), reverse=True):
        triggered = s.get('triggered', 0)
        win_pct = f'{s.get("win_rate", 0) * 100:.1f}%'
        avg_ret = f'{s.get("avg_return_pct", 0):.2f}%'
        color = 'text-green-600' if s.get('avg_return_pct', 0) > 0 else 'text-red-600'
        model_rows.append([
            _esc(model),
            str(s.get('trades', 0)),
            str(triggered),
            win_pct,
            f'<span class="{color} font-medium">{avg_ret}</span>',
        ])

    content = f'''
<h3 class="font-semibold text-gray-700 mb-2">By signal</h3>
{_table(signal_hdrs, signal_rows)}
<h3 class="font-semibold text-gray-700 mt-5 mb-2">By model</h3>
{_table(model_hdrs, model_rows)}'''

    days = report.get('days_in_history', 0)
    skipped = report.get('skipped_pairs', 0)
    return _card('Backtest Summary', content,
                 subtitle=f'{days} prediction day(s) evaluated · {skipped} ticker/date pairs skipped (no data)')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate() -> None:
    os.makedirs('docs', exist_ok=True)

    date_str, report = find_latest_signals()
    journal = load_json('history/trade_journal.json', [])
    scores = load_json(SCORES_FILE, {})
    backtest = load_json(BACKTEST_FILE, None)

    if not report and not journal and not scores:
        print('WARNING: No data found. Generating empty placeholder page.')

    # Embed build time as both a display string and an ISO timestamp for JS
    now = datetime.now(timezone.utc)
    generated_display = now.strftime('%b %d, %Y  %H:%M UTC')
    generated_iso = now.strftime('%Y-%m-%dT%H:%M:%SZ')

    signals_html = build_signals_section(report, date_str) if report else _card(
        'Signals', '<p class="text-gray-500 text-sm">No signals report found. Run forge_loop.py first.</p>'
    )
    scores_html = build_scores_section(scores)
    pnl_html = build_pnl_section(journal)
    backtest_html = build_backtest_section(backtest)

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>OracleForge</title>
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
  <div id="token-modal" class="hidden fixed inset-0 bg-black bg-opacity-40 z-50 flex items-center justify-center">
    <div class="bg-white rounded-xl shadow-xl p-6 w-full max-w-md mx-4">
      <h3 class="text-lg font-semibold text-gray-900 mb-1">GitHub Personal Access Token</h3>
      <p class="text-sm text-gray-500 mb-3">
        Required once to trigger workflow rebuilds. Stored only in your browser's localStorage —
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
        <div class="text-right hidden sm:block">
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
        <button onclick="clearToken()" title="Clear saved GitHub token"
          class="text-gray-300 hover:text-gray-500 transition-colors text-lg leading-none">⚙</button>
      </div>
    </div>
  </header>

  <!-- Nav tabs -->
  <div class="max-w-6xl mx-auto px-4 mt-4">
    <div class="flex gap-2 mb-5 border-b border-gray-200">
      <button onclick="showTab('signals')" id="tab-signals"
        class="tab-btn px-4 py-2 text-sm font-medium text-blue-600 border-b-2 border-blue-600">Signals</button>
      <button onclick="showTab('pnl')" id="tab-pnl"
        class="tab-btn px-4 py-2 text-sm font-medium text-gray-500 hover:text-gray-700 border-b-2 border-transparent">P&amp;L</button>
      <button onclick="showTab('backtest')" id="tab-backtest"
        class="tab-btn px-4 py-2 text-sm font-medium text-gray-500 hover:text-gray-700 border-b-2 border-transparent">Backtest</button>
      <button onclick="showTab('models')" id="tab-models"
        class="tab-btn px-4 py-2 text-sm font-medium text-gray-500 hover:text-gray-700 border-b-2 border-transparent">Models</button>
    </div>

    <div id="pane-signals">{signals_html}</div>
    <div id="pane-pnl" class="hidden">{pnl_html}</div>
    <div id="pane-backtest" class="hidden">{backtest_html if backtest_html else _card("Backtest", '<p class="text-gray-500 text-sm">No backtest report found. Run python backtest.py to generate one.</p>')}</div>
    <div id="pane-models" class="hidden">{scores_html}</div>
  </div>

  <script>
    // ── Tab switching ──────────────────────────────────────────────────────
    function showTab(name) {{
      ['signals','pnl','backtest','models'].forEach(t => {{
        document.getElementById('pane-' + t).classList.toggle('hidden', t !== name);
        const btn = document.getElementById('tab-' + t);
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
          // Hard reload — bypass cache so we get the newly committed HTML
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

    with open(OUT_FILE, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f'Generated {OUT_FILE} ({os.path.getsize(OUT_FILE):,} bytes)')


if __name__ == '__main__':
    generate()
