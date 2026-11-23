#!/usr/bin/env python3
"""Dispatch a trading workflow whose GitHub schedule did not fire on time.

GitHub's scheduler has fired morning_orders 5 to 9 hours late (after the
close) and the nightly 3 to 5 hours late, and it throttles the retry watchdog
too. This runs from a systemd timer on a machine we control (deploy/systemd/)
so the orders do not depend on GitHub's cron. It mirrors the watchdog in
.github/workflows/retry_failed.yml and is safe to run alongside it: a workflow
is dispatched only if no run exists since today's expected time and none is
queued or in progress, and the workflows themselves are idempotent per day.

Uses the `gh` CLI's existing login. Exit code is always 0 unless gh fails.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone

REPO = 'kdigitalsystems/OracleForge'
# wf, window start/end (UTC hours), expected fire time (hour, minute, day offset), grace hours.
SCHEDULES = [
    {'wf': 'nightly_forge.yml',  'from': 2,  'to': 13, 'expected': (23, 17, -1), 'grace_h': 1.5},
    {'wf': 'morning_orders.yml', 'from': 14, 'to': 21, 'expected': (13, 30, 0),  'grace_h': 0.75},
]


def due_time(now: datetime, expected: tuple[int, int, int]) -> datetime:
    hour, minute, day_offset = expected
    day = (now + timedelta(days=day_offset)).date()
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=timezone.utc)


def is_missed(now: datetime, latest_created: datetime | None, sched: dict) -> bool:
    """True when the scheduled run is overdue: in its window, past the grace
    period, and no run created since the expected time."""
    if now.weekday() > 4:                      # Saturday, Sunday UTC: nothing is scheduled
        return False
    if not sched['from'] <= now.hour < sched['to']:
        return False
    due = due_time(now, sched['expected'])
    if (now - due).total_seconds() / 3600 < sched['grace_h']:
        return False
    return latest_created is None or latest_created < due


def _gh(*args: str) -> str:
    return subprocess.run(['gh', *args], check=True, capture_output=True, text=True).stdout


def _runs(wf: str, **params: str) -> list[dict]:
    query = '&'.join(f'{k}={v}' for k, v in {'per_page': '1', **params}.items())
    out = _gh('api', f'repos/{REPO}/actions/workflows/{wf}/runs?{query}')
    return json.loads(out).get('workflow_runs', [])


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace('Z', '+00:00'))


def main(now: datetime | None = None, dry_run: bool = False) -> int:
    now = now or datetime.now(timezone.utc)
    for sched in SCHEDULES:
        wf = sched['wf']
        latest = _runs(wf)
        created = _parse(latest[0]['created_at']) if latest else None
        if not is_missed(now, created, sched):
            continue
        if any(_runs(wf, status=s) for s in ('queued', 'in_progress')):
            print(f'{wf}: overdue but a run is queued or in progress; no action.')
            continue
        print(f'{now:%Y-%m-%d %H:%M}Z {wf}: schedule missed (latest run {created}); dispatching.')
        if not dry_run:
            _gh('workflow', 'run', wf, '--repo', REPO, '--ref', 'main')
    return 0


if __name__ == '__main__':
    sys.exit(main(dry_run='--dry-run' in sys.argv))
