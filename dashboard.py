# dashboard.py — Streamlit UI (run: streamlit run dashboard.py)
"""OracleForge dashboard: signals, positions, trades, and model scores."""
from __future__ import annotations

import json
import os
from datetime import date

import pandas as pd
import streamlit as st

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


@st.cache_data(ttl=30)
def load_trade_log(date_str: str) -> list[dict]:
    path = os.path.join(REPORTS_DIR, f'trades_{date_str}.json')
    return load_json(path, [])


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
    today_str = date.today().strftime('%Y-%m-%d')
    trade_log = load_trade_log(today_str)

    tab_signals, tab_positions, tab_trades, tab_models = st.tabs(
        ['Signals', 'Positions', 'Trades', 'Model scores']
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
        st.subheader(f'Trade log — {today_str}')
        if not trade_log:
            st.info('No trades logged today. Run `python trader.py` during market hours.')
        else:
            trades_df = pd.DataFrame(trade_log)
            st.dataframe(trades_df, use_container_width=True, hide_index=True)
            buys = [t for t in trade_log if t.get('action') == 'BUY']
            sells = [t for t in trade_log if t.get('action') == 'SELL']
            cols = st.columns(2)
            cols[0].metric('Buys today', len(buys), f"${sum(t.get('amount_usd', 0) for t in buys):.2f}")
            cols[1].metric('Sells today', len(sells), f"${sum(t.get('amount_usd', 0) for t in sells):.2f}")

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
