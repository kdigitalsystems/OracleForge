import unittest
from datetime import datetime, timezone

from scripts.dispatch_missed import SCHEDULES, is_missed

MORNING = next(s for s in SCHEDULES if s['wf'] == 'morning_orders.yml')
NIGHTLY = next(s for s in SCHEDULES if s['wf'] == 'nightly_forge.yml')


def utc(*a):
    return datetime(*a, tzinfo=timezone.utc)


class IsMissedTests(unittest.TestCase):
    def test_yesterdays_late_run_does_not_count(self):
        # 2026-11-24 is a Tuesday; last run was Monday's late one.
        self.assertTrue(is_missed(utc(2026, 11, 24, 14, 20), utc(2026, 11, 23, 21, 50), MORNING))

    def test_waits_for_grace_period(self):
        self.assertFalse(is_missed(utc(2026, 11, 24, 13, 50), utc(2026, 11, 23, 21, 50), MORNING))

    def test_todays_run_exists(self):
        self.assertFalse(is_missed(utc(2026, 11, 24, 15, 0), utc(2026, 11, 24, 13, 41), MORNING))

    def test_no_dispatch_after_window_or_on_weekend(self):
        self.assertFalse(is_missed(utc(2026, 11, 24, 21, 30), None, MORNING))
        self.assertFalse(is_missed(utc(2026, 11, 28, 15, 0), None, MORNING))   # Saturday

    def test_never_ran_counts_as_missed(self):
        self.assertTrue(is_missed(utc(2026, 11, 24, 15, 0), None, MORNING))

    def test_nightly_uses_previous_evening_slot(self):
        # Tuesday 03:00 UTC: Monday-evening 23:17 run has not happened yet.
        self.assertTrue(is_missed(utc(2026, 11, 24, 3, 0), utc(2026, 11, 23, 3, 43), NIGHTLY))
        self.assertFalse(is_missed(utc(2026, 11, 24, 4, 0), utc(2026, 11, 24, 3, 30), NIGHTLY))
        self.assertFalse(is_missed(utc(2026, 11, 24, 0, 30), utc(2026, 11, 23, 3, 43), NIGHTLY))  # before window


if __name__ == '__main__':
    unittest.main()
