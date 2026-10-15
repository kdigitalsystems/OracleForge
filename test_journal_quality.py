"""Unit tests for journal_quality: which journal rows can be trusted."""
import unittest

from journal_quality import flagged_summary, journal_row_issue, split_journal

GOOD = {'ticker': 'AAA', 'entry_price': 100.0, 'exit_price': 105.0, 'pnl_pct': 5.0,
        'pnl_usd': 0.1, 'usd_invested': 2.0, 'outcome': 'win'}


class JournalRowIssueTests(unittest.TestCase):

    def test_healthy_row_passes(self):
        self.assertIsNone(journal_row_issue(GOOD))

    def test_desync_flag_wins(self):
        self.assertEqual(journal_row_issue({**GOOD, 'qty_desync_corrected': True}), 'qty_desync')

    def test_sign_mismatch_between_prices_and_recorded_return(self):
        # BSX 2026-05-22: bought 56.608, sold 58.09, recorded -48.69%.
        row = {**GOOD, 'entry_price': 56.608, 'exit_price': 58.09, 'pnl_pct': -48.69}
        self.assertEqual(journal_row_issue(row), 'sign_mismatch')

    def test_sign_disagreement_near_zero_is_rounding_not_a_defect(self):
        row = {**GOOD, 'entry_price': 100.0, 'exit_price': 100.02, 'pnl_pct': -0.03}
        self.assertIsNone(journal_row_issue(row))

    def test_near_zero_invested(self):
        self.assertEqual(journal_row_issue({**GOOD, 'usd_invested': 0.0, 'outcome': 'loss'}), 'near_zero_invested')

    def test_unparseable_or_missing_pnl(self):
        self.assertEqual(journal_row_issue({**GOOD, 'pnl_pct': 'n/a'}), 'unparseable')
        self.assertEqual(journal_row_issue({k: v for k, v in GOOD.items() if k != 'pnl_pct'}), 'unparseable')

    def test_row_without_optional_fields_is_judged_on_what_it_has(self):
        self.assertIsNone(journal_row_issue({'ticker': 'AAA', 'pnl_pct': 2.0}))


class SplitJournalTests(unittest.TestCase):

    def test_split_keeps_order_and_summarizes_flagged_rows(self):
        a, b = {**GOOD, 'ticker': 'A'}, {**GOOD, 'ticker': 'B', 'pnl_usd': -1.5, 'qty_desync_corrected': True}
        c, d = {**GOOD, 'ticker': 'C'}, {**GOOD, 'ticker': 'D', 'pnl_usd': -0.5, 'usd_invested': 0.0}
        reliable, flagged = split_journal([a, b, c, d])
        self.assertEqual([t['ticker'] for t in reliable], ['A', 'C'])
        self.assertEqual([t['ticker'] for t in flagged], ['B', 'D'])
        self.assertEqual(flagged_summary(flagged),
                         {'count': 2, 'recorded_pnl_usd': -2.0, 'reasons': {'qty_desync': 1, 'near_zero_invested': 1}})

    def test_empty_journal(self):
        self.assertEqual(split_journal([]), ([], []))
        self.assertEqual(flagged_summary([]), {'count': 0, 'recorded_pnl_usd': 0.0, 'reasons': {}})


if __name__ == '__main__':
    unittest.main()
