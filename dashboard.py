# dashboard.py — Streamlit UI (run: streamlit run dashboard.py)
"""OracleForge dashboard: signals, positions, trades, and model scores."""
from __future__ import annotations

import json
import os
from datetime import datetime
from collections import defaultdict

import pandas as pd
import pytz
import streamlit as st

# Match trader.py / forge_loop.py: "today" is the US market day, not the local clock.
ET = pytz.timezone('America/New_York')
OPEN_ORDERS_FILE = 'state/open_orders.json'

from signals import (
    HISTORY_DIR,
    REPORTS_DIR,
    SCORES_FILE,
    generate_report_from_history,
    list_prediction_dates,
    load_json,
    load_signal_config,
)

st.set_page_config(page_title='OracleForge', layout='wide')


@st.cache_data(ttl=60)
def load_report(date_str: str) -> dict | None:
    path = os.path.join(REPORTS_DIR, f'signals_{date_str}.json')
    if os.path.exists(path):
        return load_json(path, None)
    try:
        _, report = generate_report_from_history(date_str)
        return report
    except Exception:
        return None


@st.cache_data(ttl=60)
def load_predictions(date_str: str) -> dict:
    path = os.path.join(HISTORY_DIR, f'predictions_{date_str}.json')
    return load_json(path, {})


TRADE_JOURNAL_FILE = 'history/trade_journal.json'


@st.cache_data(ttl=60)
def load_trade_journal() -> list[dict]:
    return load_json(TRADE_JOURNAL_FILE, [])


@st.cache_data(ttl=30)
def load_open_orders() -> dict:
    """Tracked open orders shared between the --open and --close trader jobs."""
    return load_json(OPEN_ORDERS_FILE, {})


@st.cache_data(ttl=30)
def load_alpaca_positions() -> dict[str, float] | None:
    """Returns {symbol: market_value} or None if unavailable."""
    try:
        import alpaca_client
        client = alpaca_client.get_trading_client()
        return alpaca_client.get_positions(client)
    except Exception:
        return None


def main() -> None:
    st.title('OracleForge')
    st.caption('LLM ensemble — overnight buy/sell range predictions with Alpaca paper trading.')

    dates = list_prediction_dates()
    if not dates:
        st.warning('No prediction history found. Run `python forge_loop.py` first.')
        st.stop()

    sidebar = st.sidebar
    sidebar.header('Settings')
    selected_date = sidebar.selectbox('Prediction date', options=list(reversed(dates)), index=0)
    signal_config = load_signal_config()
    sidebar.json(signal_config, expanded=False)

    report = load_report(selected_date)
    predictions = load_predictions(selected_date)
    scores = load_json(SCORES_FILE, {})
    today_str = datetime.now(ET).strftime('%Y-%m-%d')

    tab_signals, tab_positions, tab_trades, tab_pnl, tab_models = st.tabs(
        ['Signals', 'Positions', 'Trades', 'P&L', 'Model scores']
    )

    with tab_signals:
        st.subheader(f'Signals — {selected_date}')
        if report:
            summary = report.get('summary', {})
            cols = st.columns(4)
            cols[0].metric('Total', summary.get('total', 0))
            cols[1].metric('ACTIVE', summary.get('active', 0))
            cols[2].metric('SKIP', summary.get('skip', 0))
            cols[3].metric('STALE', summary.get('stale', 0))

            active_rows = report.get('active', [])
            if not active_rows:
                st.info('No ACTIVE setups for this date.')
            else:
                df = pd.DataFrame(active_rows)
                st.dataframe(df, use_container_width=True, hide_index=True)
                if 'upside_pct' in df.columns:
                    st.bar_chart(df.set_index('ticker')['upside_pct'])

            with st.expander('All tickers'):
                all_rows = (
                    report.get('active', [])
                    + report.get('skip', [])
                    + report.get('stale', [])
                )
                if all_rows:
                    all_df = pd.DataFrame(all_rows)
                    signal_filter = st.multiselect(
                        'Filter by signal',
                        options=['ACTIVE', 'SKIP', 'STALE'],
                        default=['ACTIVE', 'SKIP', 'STALE'],
                        key='signal_filter',
                    )
                    filtered = all_df[all_df['signal'].isin(signal_filter)] if signal_filter else all_df
                    st.dataframe(filtered, use_container_width=True, hide_index=True)
        else:
            st.info('No signals report. Run `python signals.py` or `python forge_loop.py`.')

        # Per-ticker model breakdown
        if predictions:
            ticker_options = sorted(predictions.keys())
            selected_ticker = st.selectbox('Model detail for ticker', options=['—'] + ticker_options)
            if selected_ticker and selected_ticker != '—' and selected_ticker in predictions:
                entry = predictions[selected_ticker]
                st.write(f"**Close:** ${entry.get('close', '—')}")
                consensus = entry.get('consensus') or {}
                if consensus:
                    st.write(
                        f"**Consensus buy range:** ${consensus.get('buy_low')} – ${consensus.get('buy_high')}"
                    )
                    st.write(
                        f"**Consensus sell range:** ${consensus.get('sell_low')} – ${consensus.get('sell_high')}"
                    )
                models = entry.get('models', {})
                if models:
                    model_rows = []
                    for mname, mdata in models.items():
                        if isinstance(mdata, dict):
                            model_rows.append({
                                'model': mname,
                                'buy_low': mdata.get('buy_low'),
                                'buy_high': mdata.get('buy_high'),
                                'sell_low': mdata.get('sell_low'),
                                'sell_high': mdata.get('sell_high'),
                                'rationale': mdata.get('rationale', ''),
                            })
                    if model_rows:
                        st.dataframe(pd.DataFrame(model_rows), use_container_width=True, hide_index=True)

    with tab_positions:
        st.subheader('Live Alpaca paper positions')
        if st.button('Refresh positions'):
            st.cache_data.clear()

        positions = load_alpaca_positions()
        if positions is None:
            st.warning('Could not connect to Alpaca. Check `~/.ssh/alpaca_paper_keys`.')
        elif not positions:
            st.info('No open positions.')
        else:
            pos_df = pd.DataFrame(
                [{'symbol': sym, 'market_value_usd': val} for sym, val in positions.items()]
            ).sort_values('market_value_usd', ascending=False)
            st.dataframe(pos_df, use_container_width=True, hide_index=True)
            st.metric('Total portfolio value', f"${sum(positions.values()):.2f}")

    with tab_trades:
        st.subheader('Order activity')

        # Tracked open orders (state shared between --open and --close jobs)
        open_orders = load_open_orders()
        active_orders = {
            tk: e for tk, e in open_orders.items()
            if isinstance(e, dict) and not e.get('closed')
        }
        st.markdown('**Tracked open orders** — `state/open_orders.json`')
        if not active_orders:
            st.info('No tracked open orders. Run `python trader.py --open` during market hours.')
        else:
            order_rows = []
            for tk, e in active_orders.items():
                order_rows.append({
                    'ticker': tk,
                    'qty': e.get('qty'),
                    'buy_limit': e.get('buy_limit'),
                    'sell_limit': e.get('sell_limit'),
                    'stop_limit': e.get('stop_limit'),
                    'buy': 'open' if e.get('buy_order_id') else '—',
                    'sell': 'open' if e.get('sell_order_id') else '—',
                    'stop': 'open' if e.get('stop_order_id') else '—',
                    'placed': e.get('date'),
                })
            st.dataframe(pd.DataFrame(order_rows), use_container_width=True, hide_index=True)

        # Positions closed today (from the trade journal)
        st.markdown(f'**Closed today — {today_str}**')
        journal = load_trade_journal()
        todays = [t for t in journal if t.get('close_date') == today_str]
        if not todays:
            st.info('No positions closed today yet.')
        else:
            tdf = pd.DataFrame(todays)
            st.dataframe(
                tdf[['ticker', 'entry_price', 'exit_price', 'usd_invested',
                     'usd_returned', 'pnl_usd', 'pnl_pct', 'outcome']],
                use_container_width=True, hide_index=True,
            )
            wins = sum(1 for t in todays if t.get('outcome') == 'win')
            cols = st.columns(3)
            cols[0].metric('Closed today', len(todays))
            cols[1].metric('Wins', wins)
            cols[2].metric("Today's P&L", f"${sum(t.get('pnl_usd', 0) for t in todays):+.4f}")

    with tab_pnl:
        st.subheader('P&L — trade journal')
        journal = load_trade_journal()
        if not journal:
            st.info('No closed trades yet. P&L will appear here after trader.py closes positions.')
        else:
            jdf = pd.DataFrame(journal)
            jdf['close_date'] = pd.to_datetime(jdf['close_date'])

            # Summary metrics
            total_pnl = jdf['pnl_usd'].sum()
            wins = (jdf['outcome'] == 'win').sum()
            win_rate = wins / len(jdf) * 100
            avg_pnl_pct = jdf['pnl_pct'].mean()

            mcols = st.columns(4)
            mcols[0].metric('Total trades', len(jdf))
            mcols[1].metric('Win rate', f'{win_rate:.0f}%')
            mcols[2].metric('Total P&L', f'${total_pnl:+.4f}')
            mcols[3].metric('Avg P&L %', f'{avg_pnl_pct:+.2f}%')

            # Cumulative P&L curve
            st.subheader('Cumulative P&L over time')
            daily_pnl = jdf.groupby('close_date')['pnl_usd'].sum().sort_index().cumsum()
            st.line_chart(daily_pnl.rename('Cumulative P&L ($)'))

            # Per-model win rate
            st.subheader('Win rate by model')
            model_stats: dict[str, dict] = defaultdict(lambda: {'wins': 0, 'total': 0, 'pnl': 0.0})
            for _, row in jdf.iterrows():
                for model in row.get('predicting_models') or []:
                    model_stats[model]['total'] += 1
                    model_stats[model]['pnl'] += row['pnl_usd']
                    if row['outcome'] == 'win':
                        model_stats[model]['wins'] += 1
            if model_stats:
                model_rows = [
                    {
                        'model': m,
                        'trades': s['total'],
                        'win_rate_pct': round(s['wins'] / s['total'] * 100, 1) if s['total'] else 0,
                        'total_pnl_usd': round(s['pnl'], 4),
                    }
                    for m, s in model_stats.items()
                ]
                st.dataframe(
                    pd.DataFrame(model_rows).sort_values('win_rate_pct', ascending=False),
                    use_container_width=True, hide_index=True,
                )

            # Best / worst trades
            col_best, col_worst = st.columns(2)
            with col_best:
                st.subheader('Best trades')
                st.dataframe(
                    jdf.nlargest(5, 'pnl_pct')[
                        ['close_date', 'ticker', 'entry_price', 'exit_price', 'pnl_usd', 'pnl_pct']
                    ].reset_index(drop=True),
                    use_container_width=True, hide_index=True,
                )
            with col_worst:
                st.subheader('Worst trades')
                st.dataframe(
                    jdf.nsmallest(5, 'pnl_pct')[
                        ['close_date', 'ticker', 'entry_price', 'exit_price', 'pnl_usd', 'pnl_pct']
                    ].reset_index(drop=True),
                    use_container_width=True, hide_index=True,
                )

            # Full journal
            with st.expander('Full trade journal'):
                st.dataframe(
                    jdf.sort_values('close_date', ascending=False).reset_index(drop=True),
                    use_container_width=True, hide_index=True,
                )

    with tab_models:
        st.subheader('Analyst scores (MoE weights)')
        if not scores:
            st.warning('No scores in state/analyst_scores.json')
        else:
            score_df = pd.DataFrame(
                [{'model': k, 'score': v} for k, v in scores.items()]
            ).sort_values('score', ascending=False)
            st.dataframe(score_df, use_container_width=True, hide_index=True)
            st.bar_chart(score_df.set_index('model')['score'])


if __name__ == '__main__':
    main()
