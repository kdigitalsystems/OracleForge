"""Which trade-journal rows can be trusted for P&L statistics.

Rows closed before the qty-desync fix (2026-07/08) can carry a cost basis that
disagrees with the shares actually sold, e.g. a BSX trade bought at 56.61 and
sold at 58.09 recorded as -48.7%. They stay in history/trade_journal.json
untouched; readers (the dashboard, trade attribution, backtest calibration)
use split_journal() to keep them out of headline numbers and to say how many
were left out and what they sum to as recorded.
"""
from __future__ import annotations

MIN_INVESTED_USD = 0.25    # below this the percentages are rounding noise
SIGN_TOLERANCE_PCT = 0.1   # ignore sign disagreements this close to zero
MISMATCH_PCT = 1.0         # recorded vs price-implied return gap that counts


def journal_row_issue(trade: dict) -> str | None:
    """Return why a row is unreliable, or None if it can be trusted."""
    if trade.get('qty_desync_corrected'):
        return 'qty_desync'
    def number(key):
        value = trade.get(key)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ValueError(key)

    try:
        pct = number('pnl_pct')
        invested, entry, exit_ = number('usd_invested'), number('entry_price'), number('exit_price')
    except ValueError:
        return 'unparseable'
    if pct is None:
        return 'unparseable'
    # The remaining checks need optional fields; a row without them is judged
    # on what it does have.
    if invested is not None and invested < MIN_INVESTED_USD:
        return 'near_zero_invested'
    if entry and exit_ and entry > 0 and exit_ > 0:
        implied = (exit_ / entry - 1) * 100
        if (abs(implied) > SIGN_TOLERANCE_PCT and abs(pct) > SIGN_TOLERANCE_PCT
                and (implied > 0) != (pct > 0) and abs(implied - pct) > MISMATCH_PCT):
            return 'sign_mismatch'
    return None


def split_journal(journal: list[dict]) -> tuple[list[dict], list[dict]]:
    """(reliable rows, flagged rows), each in the journal's original order."""
    reliable, flagged = [], []
    for trade in journal:
        (flagged if journal_row_issue(trade) else reliable).append(trade)
    return reliable, flagged


def flagged_summary(flagged: list[dict]) -> dict:
    """Count and as-recorded dollar total of the excluded rows, plus reasons."""
    reasons: dict[str, int] = {}
    total = 0.0
    for trade in flagged:
        reason = journal_row_issue(trade) or 'unknown'
        reasons[reason] = reasons.get(reason, 0) + 1
        try:
            total += float(trade.get('pnl_usd') or 0)
        except (TypeError, ValueError):
            pass
    return {'count': len(flagged), 'recorded_pnl_usd': round(total, 2), 'reasons': reasons}
