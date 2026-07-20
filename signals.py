# signals.py
"""Score-weighted consensus and daily buy/sell range signals."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Any

CONFIG_FILE = 'config/signals.json'
HISTORY_DIR = 'history/'
REPORTS_DIR = 'reports/'
SCORES_FILE = 'state/analyst_scores.json'

RANGE_FIELDS = ('buy_low', 'buy_high', 'sell_low', 'sell_high')

DEFAULT_SIGNAL_CONFIG = {
    'min_upside_pct': 1.0,
    'min_range_width_pct': 0.5,
    'min_agreeing_models': 2,
}


def load_json(filepath: str, default: Any) -> Any:
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return default


def save_json(filepath: str, data: Any) -> None:
    # Write-then-rename: os.replace is atomic on POSIX and Windows (same
    # filesystem), so a reader/writer racing this call always sees either the
    # complete old file or the complete new one -- never a torn/interleaved
    # write from two processes writing the same path concurrently (seen in
    # practice when two runs land on the same self-hosted machine).
    directory = os.path.dirname(filepath) or '.'
    os.makedirs(directory, exist_ok=True)
    tmp_path = f'{filepath}.tmp{os.getpid()}'
    with open(tmp_path, 'w') as f:
        json.dump(data, f, indent=4)
    os.replace(tmp_path, filepath)


def load_signal_config() -> dict:
    config = load_json(CONFIG_FILE, {})
    return {**DEFAULT_SIGNAL_CONFIG, **config}


def parse_ticker_list(tickers_arg: str | None) -> list[str] | None:
    if not tickers_arg:
        return None
    tickers = [part.strip().upper() for part in tickers_arg.split(',') if part.strip()]
    return tickers or None


def list_prediction_dates() -> list[str]:
    if not os.path.isdir(HISTORY_DIR):
        return []
    dates = []
    for name in os.listdir(HISTORY_DIR):
        if name.startswith('predictions_') and name.endswith('.json'):
            dates.append(name.replace('predictions_', '').replace('.json', ''))
    return sorted(dates)


def extract_model_predictions(ticker_entry: Any) -> dict[str, dict]:
    """Extract per-model range dicts from a history entry."""
    if not isinstance(ticker_entry, dict):
        return {}
    models = ticker_entry.get('models', {})
    if not isinstance(models, dict):
        return {}
    return {k: v for k, v in models.items() if isinstance(v, dict) and 'buy_low' in v}


def weighted_consensus_ranges(
    model_range_preds: dict[str, dict],
    scores: dict[str, float],
    min_agreeing_models: int = 2,
    max_cv: float | None = None,
) -> dict[str, float] | None:
    """Compute score-weighted average of buy/sell ranges across models.

    Only models with a valid prediction (buy_low < buy_high < sell_low <= sell_high)
    contribute. Fallback (synthetic) predictions are excluded. Returns None if:
    - fewer than min_agreeing_models have valid ranges, or
    - model disagreement (CV) on buy_high or sell_low exceeds max_cv.
    """
    if not model_range_preds:
        return None

    total_weight = 0.0
    weighted: dict[str, float] = {f: 0.0 for f in RANGE_FIELDS}
    valid_values: list[dict] = []  # per-model field values for CV check
    valid_weights: list[float] = []
    valid_count = 0

    for model_name, ranges in model_range_preds.items():
        if not isinstance(ranges, dict):
            continue
        # Skip fallback predictions ? synthetic ranges contaminate consensus
        if ranges.get('fallback'):
            continue
        bl = float(ranges.get('buy_low') or 0)
        bh = float(ranges.get('buy_high') or 0)
        sl = float(ranges.get('sell_low') or 0)
        sh = float(ranges.get('sell_high') or 0)
        # Skip fallbacks, skipped tickers, and incoherent ranges
        if not (bl > 0 and bh > bl and sl > bh and sh >= sl):
            continue

        weight = max(float(scores.get(model_name, 5.0)), 0.1)
        for field in RANGE_FIELDS:
            weighted[field] += float(ranges[field]) * weight
        total_weight += weight
        valid_count += 1
        valid_values.append({f: float(ranges[f]) for f in RANGE_FIELDS})
        valid_weights.append(weight)

    if valid_count < min_agreeing_models or total_weight == 0:
        return None

    consensus = {f: round(weighted[f] / total_weight, 2) for f in RANGE_FIELDS}

    # Coefficient of variation = weighted model disagreement on the key levels.
    # Compute it always (so it can be surfaced on the dashboard) and, when
    # max_cv is set, reject consensus where the models disagree too much.
    field_cvs = []
    if valid_count >= 2:
        for field in ('buy_high', 'sell_low'):
            mean_val = consensus[field]
            if mean_val <= 0:
                continue
            # Weighted variance around the consensus mean
            variance = sum(
                w * (v[field] - mean_val) ** 2
                for v, w in zip(valid_values, valid_weights)
            ) / total_weight
            field_cvs.append((variance ** 0.5) / mean_val)

    consensus_cv = round(max(field_cvs), 4) if field_cvs else 0.0
    if max_cv is not None and consensus_cv > max_cv:
        return None
    consensus['consensus_cv'] = consensus_cv

    return consensus


def classify_opportunity(
    close: float,
    consensus: dict[str, float] | None,
    config: dict | None = None,
) -> dict[str, Any]:
    """Classify the trade setup for one ticker."""
    config = config or load_signal_config()
    min_upside = float(config['min_upside_pct'])

    if not consensus or close <= 0:
        return {'signal': 'SKIP', 'upside_pct': None, 'consensus': consensus}

    buy_low = consensus.get('buy_low', 0)
    buy_high = consensus.get('buy_high', 0)
    sell_low = consensus.get('sell_low', 0)
    sell_high = consensus.get('sell_high', 0)

    if buy_high <= 0 or sell_low <= buy_high:
        return {'signal': 'SKIP', 'upside_pct': None, 'consensus': consensus}

    # Upside from entry (buy_high) to target (sell_low) ? actual expected trade return
    upside_pct = round(((sell_low - buy_high) / buy_high) * 100, 2)

    # Spread filter: reject signals where the buy range is suspiciously wide
    max_spread = float(config.get('max_spread_pct', 999.0))
    buy_spread_pct = ((buy_high - buy_low) / close) * 100 if close > 0 else 999.0
    if buy_spread_pct > max_spread:
        return {'signal': 'SKIP', 'upside_pct': upside_pct, 'consensus': consensus}

    if close >= sell_low:
        signal = 'STALE'
        upside_pct = None  # opportunity already missed, upside is meaningless
    elif upside_pct < min_upside:
        signal = 'SKIP'
    else:
        signal = 'ACTIVE'

    return {
        'signal': signal,
        'upside_pct': upside_pct,
        'consensus': consensus,
    }


def build_enriched_predictions(
    raw_predictions: dict[str, dict[str, dict]],
    closes: dict[str, float],
    scores: dict[str, float],
    config: dict | None = None,
) -> dict[str, dict]:
    """Turn per-model range outputs into enriched per-ticker records."""
    config = config or load_signal_config()
    enriched = {}

    min_agreeing = int(config.get('min_agreeing_models', 2))
    max_cv_raw = config.get('max_consensus_cv')
    max_cv = float(max_cv_raw) if max_cv_raw is not None else None

    for ticker, model_preds in raw_predictions.items():
        close = closes.get(ticker)
        if close is None:
            continue

        consensus = weighted_consensus_ranges(model_preds, scores, min_agreeing, max_cv=max_cv)
        opp = classify_opportunity(close, consensus, config)

        enriched[ticker] = {
            'close': round(float(close), 2),
            'models': model_preds,
            'consensus': consensus,
            'signal': opp['signal'],
            'upside_pct': opp['upside_pct'],
        }

    return enriched


def build_signals_report(
    enriched_predictions: dict[str, dict],
    date_str: str | None = None,
) -> dict:
    date_str = date_str or datetime.now().strftime('%Y-%m-%d')
    active, skip, stale = [], [], []

    for ticker, entry in enriched_predictions.items():
        consensus = entry.get('consensus') or {}
        row = {
            'ticker': ticker,
            'close': entry.get('close'),
            'buy_low': consensus.get('buy_low'),
            'buy_high': consensus.get('buy_high'),
            'sell_low': consensus.get('sell_low'),
            'sell_high': consensus.get('sell_high'),
            'upside_pct': entry.get('upside_pct'),
            'consensus_cv': consensus.get('consensus_cv'),
            'signal': entry.get('signal'),
        }
        signal = entry.get('signal')
        if signal == 'ACTIVE':
            active.append(row)
        elif signal == 'STALE':
            stale.append(row)
        else:
            skip.append(row)

    sort_key = lambda row: row.get('upside_pct') if row.get('upside_pct') is not None else -999
    active.sort(key=sort_key, reverse=True)
    skip.sort(key=sort_key, reverse=True)

    return {
        'date': date_str,
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'config': load_signal_config(),
        'summary': {
            'total': len(enriched_predictions),
            'active': len(active),
            'skip': len(skip),
            'stale': len(stale),
        },
        'active': active,
        'skip': skip,
        'stale': stale,
    }


def print_signals_table(report: dict, limit: int = 20) -> None:
    print(f"\n=== OracleForge Signals ({report['date']}) ===")
    summary = report['summary']
    print(
        f"Total: {summary['total']} | "
        f"ACTIVE: {summary['active']} | "
        f"SKIP: {summary['skip']} | "
        f"STALE: {summary['stale']}"
    )

    rows = report['active'][:limit]
    if not rows:
        print("\nNo ACTIVE setups today.")
        return

    print(f"\nTop {len(rows)} ACTIVE setups:")
    print(
        f"{'Ticker':<8} {'Close':>8} {'Buy Low':>9} {'Buy High':>9} "
        f"{'Sell Low':>9} {'Sell High':>10} {'Upside%':>9}"
    )
    print('-' * 72)
    for row in rows:
        print(
            f"{row['ticker']:<8} "
            f"{row['close']:>8.2f} "
            f"{row['buy_low']:>9.2f} "
            f"{row['buy_high']:>9.2f} "
            f"{row['sell_low']:>9.2f} "
            f"{row['sell_high']:>10.2f} "
            f"{row['upside_pct']:>8.2f}%"
        )


def find_predictions_path(date_str: str | None) -> str | None:
    if date_str:
        path = os.path.join(HISTORY_DIR, f'predictions_{date_str}.json')
        return path if os.path.exists(path) else None

    if not os.path.isdir(HISTORY_DIR):
        return None
    candidates = sorted(
        (
            os.path.join(HISTORY_DIR, name)
            for name in os.listdir(HISTORY_DIR)
            if name.startswith('predictions_') and name.endswith('.json')
        ),
        reverse=True,
    )
    return candidates[0] if candidates else None


def generate_report_from_history(
    date_str: str | None = None,
    scores: dict | None = None,
    config: dict | None = None,
) -> tuple[dict, dict]:
    path = find_predictions_path(date_str)
    if not path:
        raise FileNotFoundError('No predictions file found in history/')

    predictions = load_json(path, {})
    scores = scores or load_json(SCORES_FILE, {})
    config = config or load_signal_config()

    closes = {}
    raw = {}
    for ticker, entry in predictions.items():
        if not isinstance(entry, dict):
            continue
        if 'close' in entry:
            closes[ticker] = float(entry['close'])
        model_preds = extract_model_predictions(entry)
        if model_preds:
            raw[ticker] = model_preds

    if not closes:
        raise ValueError('No close prices in predictions file.')

    enriched = build_enriched_predictions(raw, closes, scores, config)
    file_date = os.path.basename(path).replace('predictions_', '').replace('.json', '')
    report = build_signals_report(enriched, file_date)
    return enriched, report


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate OracleForge daily signals report.')
    parser.add_argument('--date', help='Date YYYY-MM-DD (default: latest predictions file)')
    parser.add_argument('--limit', type=int, default=20, help='Max ACTIVE rows to print')
    parser.add_argument('--min-upside', type=float, help='Override min upside %% threshold')
    args = parser.parse_args()

    config = load_signal_config()
    if args.min_upside is not None:
        config['min_upside_pct'] = args.min_upside

    enriched, report = generate_report_from_history(args.date, config=config)

    date_str = report['date']
    report_path = os.path.join(REPORTS_DIR, f'signals_{date_str}.json')
    save_json(report_path, report)
    print_signals_table(report, limit=args.limit)
    print(f"\nReport saved to {report_path}")


if __name__ == '__main__':
    main()
