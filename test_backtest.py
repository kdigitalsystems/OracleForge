"""Unit tests for the live-rules backtest."""
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import backtest
from backtest import (
    MIN_ADEQUATE_SAMPLE,
    _grid_candidates,
    _stats,
    build_signal_days,
    calibrate_against_journal,
    consensus_source,
    simulate_position,
    simulate_strategy,
    summarize_trade_attribution,
    summarize_trades,
    walk_forward_optimize,
)

# Mon 2026-01-05 .. Fri 2026-01-16 (no holidays), so calendar-day max-hold
# arithmetic is easy to follow.
SESSIONS = ['2026-01-05', '2026-01-06', '2026-01-07', '2026-01-08', '2026-01-09',
            '2026-01-12', '2026-01-13', '2026-01-14', '2026-01-15', '2026-01-16']


def _bars(*days):
    """Bars for consecutive sessions from SESSIONS[0]: (open, high, low, close)."""
    return {SESSIONS[i]: dict(zip(('open', 'high', 'low', 'close'), d)) for i, d in enumerate(days)}


FLAT = (100, 101, 99, 100)


def _sim(bars, buy_high=100.0, sell_low=105.0, stop=0.95, max_hold=15,
         signal_date='2026-01-02', entry_idx=0, end_idx=None):
    return simulate_position(bars, SESSIONS, entry_idx, signal_date, buy_high, sell_low,
                             stop, max_hold, end_idx)


class SimulatePositionTests(unittest.TestCase):

    def test_no_fill_when_session_stays_above_buy_high(self):
        self.assertIsNone(_sim(_bars((102, 104, 100.5, 103))))

    def test_gap_down_fills_at_the_open_not_the_limit(self):
        t = _sim(_bars((98, 99, 97, 98.5), (99, 106, 98, 104)))
        self.assertEqual(t['entry_price'], 98)

    def test_target_cannot_fill_on_the_entry_day(self):
        # Live places the profit-target sell only after the entry day's close.
        t = _sim(_bars((101, 107, 99.5, 106), (104, 106.5, 103, 105)))
        self.assertEqual(t['exit_date'], SESSIONS[1])
        self.assertEqual(t['exit_reason'], 'target')
        self.assertEqual(t['exit_price'], 105)

    def test_gap_up_through_target_fills_at_the_open(self):
        t = _sim(_bars(FLAT, (108, 109, 107, 108)))
        self.assertEqual(t['exit_price'], 108)

    def test_stop_on_entry_day_close_fills_at_next_open(self):
        t = _sim(_bars((100, 100.5, 94, 94.5), (93, 96, 92, 95)))
        self.assertEqual(t['exit_reason'], 'stop')
        self.assertEqual((t['exit_date'], t['exit_price']), (SESSIONS[1], 93))
        # The live journal books the close the check saw instead.
        self.assertEqual((t['booked_exit_date'], t['booked_exit_price']), (SESSIONS[0], 94.5))
        self.assertAlmostEqual(t['return_pct'], -7.0)
        self.assertAlmostEqual(t['booked_return_pct'], -5.5)

    def test_intraday_dip_below_stop_does_not_stop_if_close_recovers(self):
        t = _sim(_bars((100, 101, 93, 99), (99, 106, 98, 105)))
        self.assertEqual(t['exit_reason'], 'target')

    def test_max_hold_counts_calendar_days_from_the_signal_date(self):
        # Signal Fri 01-02, max 7 days: the Fri 01-09 close is day 7.
        t = _sim(_bars(*[FLAT] * 6), max_hold=7)
        self.assertEqual(t['exit_reason'], 'max_hold')
        self.assertEqual(t['booked_exit_date'], '2026-01-09')
        self.assertEqual(t['exit_date'], '2026-01-12')

    def test_still_held_at_end_is_open_and_valued_at_last_close(self):
        t = _sim(_bars(FLAT, (100, 102, 99, 101)))
        self.assertEqual(t['exit_reason'], 'open')
        self.assertEqual(t['exit_price'], 101)

    def test_end_idx_truncates_the_replay(self):
        bars = _bars(FLAT, FLAT, (104, 106, 103, 105))
        self.assertEqual(_sim(bars, end_idx=1)['exit_reason'], 'open')
        self.assertEqual(_sim(bars)['exit_reason'], 'target')


def _day(signal_date, entry_session, **tickers):
    return {'signal_date': signal_date, 'entry_session': entry_session, 'predictions': {
        t: {'signal': sig, 'upside_pct': 5.0, 'consensus': {'buy_high': 100.0, 'sell_low': 105.0}}
        for t, sig in tickers.items()
    }}


class SimulateStrategyTests(unittest.TestCase):

    def test_held_ticker_is_not_bought_again(self):
        bars = {'AAA': _bars(FLAT, FLAT, FLAT, (100, 106, 99, 105), FLAT)}
        days = [_day('2026-01-02', SESSIONS[0], AAA='ACTIVE'),
                _day('2026-01-05', SESSIONS[1], AAA='ACTIVE')]
        out = simulate_strategy(days, bars, SESSIONS, consensus_source(), 0.95, 15)
        self.assertEqual(len(out['trades']), 1)
        self.assertEqual(out['skipped_already_held'], 1)

    def test_rebuy_allowed_on_stop_exit_session_not_on_target_session(self):
        stop = {'AAA': _bars((100, 100, 94, 94), (95, 99, 94, 98), FLAT)}
        days = [_day('2026-01-02', SESSIONS[0], AAA='ACTIVE'),
                _day('2026-01-05', SESSIONS[1], AAA='ACTIVE')]
        # Stopped at the 01-05 close, sold at the 01-06 open: tracking was
        # cleared that evening, so the 01-06 morning can buy again.
        self.assertEqual(len(simulate_strategy(days, stop, SESSIONS, consensus_source(), 0.95, 15)['trades']), 2)
        target = {'AAA': _bars(FLAT, (100, 106, 99, 105), FLAT)}
        out = simulate_strategy(days, target, SESSIONS, consensus_source(), 0.95, 15)
        # Target fills during 01-06, so the position is still tracked that morning.
        self.assertEqual((len(out['trades']), out['skipped_already_held']), (1, 1))

    def test_only_stored_active_signals_trade(self):
        bars = {'AAA': _bars(FLAT, FLAT), 'BBB': _bars(FLAT, FLAT)}
        days = [_day('2026-01-02', SESSIONS[0], AAA='ACTIVE', BBB='SKIP')]
        trades = simulate_strategy(days, bars, SESSIONS, consensus_source(), 0.95, 15)['trades']
        self.assertEqual([t['ticker'] for t in trades], ['AAA'])

    def test_min_upside_can_only_tighten(self):
        days = [_day('2026-01-02', SESSIONS[0], AAA='ACTIVE')]
        days[0]['predictions']['AAA']['upside_pct'] = 2.0
        bars = {'AAA': _bars(FLAT, FLAT)}
        self.assertEqual(len(simulate_strategy(days, bars, SESSIONS, consensus_source(3.0), 0.95, 15)['trades']), 0)
        self.assertEqual(len(simulate_strategy(days, bars, SESSIONS, consensus_source(1.5), 0.95, 15)['trades']), 1)


class BuildSignalDaysTests(unittest.TestCase):

    def _build(self, files):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, 'history'))
            os.makedirs(os.path.join(d, 'reports'))
            for date, generated in files.items():
                with open(os.path.join(d, 'history', f'predictions_{date}.json'), 'w') as f:
                    json.dump({}, f)
                with open(os.path.join(d, 'reports', f'signals_{date}.json'), 'w') as f:
                    json.dump({'generated_at': generated}, f)
            with patch.object(backtest, 'HISTORY_DIR', os.path.join(d, 'history')), \
                    patch.object(backtest, 'REPORTS_DIR', os.path.join(d, 'reports')):
                return {x['signal_date']: x['entry_session'] for x in build_signal_days(sorted(files), SESSIONS)}

    def test_evening_and_weekend_files_trade_the_next_session(self):
        days = self._build({'2026-01-06': '2026-01-06T23:40:00', '2026-01-11': '2026-01-12T00:30:00'})
        self.assertEqual(days, {'2026-01-06': '2026-01-07', '2026-01-11': '2026-01-12'})

    def test_file_generated_early_on_its_own_date_trades_that_session(self):
        # Pre-fix runs started after midnight dated themselves by the new day.
        days = self._build({'2026-01-07': '2026-01-07T03:05:00'})
        self.assertEqual(days, {'2026-01-07': '2026-01-07'})

    def test_newest_file_wins_when_two_map_to_one_session(self):
        days = self._build({'2026-01-09': '2026-01-09T23:00:00', '2026-01-11': '2026-01-11T23:00:00'})
        self.assertEqual(days, {'2026-01-11': '2026-01-12'})


def _trade(ret, reason='target', ticker='AAA', signal='2026-01-02', booked=None):
    return {'ticker': ticker, 'signal_date': signal, 'exit_reason': reason, 'return_pct': ret,
            'booked_return_pct': ret if booked is None else booked, 'sessions_held': 2}


class SummarizeTradesTests(unittest.TestCase):

    def test_stats_exclude_open_positions(self):
        s = summarize_trades([_trade(4.0), _trade(-6.0, 'stop'), _trade(1.0, 'open')])
        self.assertEqual((s['trades'], s['open_positions']), (2, 1))
        self.assertEqual(s['win_rate'], 0.5)
        self.assertAlmostEqual(s['profit_factor'], 4 / 6, places=3)
        self.assertEqual(s['by_exit_reason']['stop']['trades'], 1)
        self.assertAlmostEqual(s['total_pnl_usd'], 2.0 * (4 - 6) / 100, places=2)

    def test_booked_key_scores_journal_prices(self):
        s = summarize_trades([_trade(-7.0, 'stop', booked=-5.5)], key='booked_return_pct')
        self.assertEqual(s['avg_return_pct'], -5.5)

    def test_max_consecutive_losses(self):
        s = summarize_trades([_trade(-1), _trade(-2), _trade(3), _trade(-1)])
        self.assertEqual(s['max_consecutive_losses'], 2)


class CalibrationTests(unittest.TestCase):

    def test_matches_on_ticker_and_signal_date_and_skips_desync_rows(self):
        sim = [_trade(3.0, ticker='AAA'), _trade(-5.5, 'stop', ticker='BBB'), _trade(2.0, ticker='CCC')]
        journal = [
            {'ticker': 'AAA', 'entry_date': '2026-01-02', 'pnl_pct': 4.0},
            {'ticker': 'BBB', 'entry_date': '2026-01-02', 'pnl_pct': 1.0},
            {'ticker': 'DDD', 'entry_date': '2026-01-02', 'pnl_pct': 1.0},
            {'ticker': 'CCC', 'entry_date': '2026-01-02', 'pnl_pct': -60.0, 'qty_desync_corrected': True},
        ]
        c = calibrate_against_journal(sim, journal)
        self.assertEqual((c['matched'], c['live_only'], c['simulated_only']), (2, 1, 1))
        self.assertEqual(c['same_win_loss'], 1)


class WalkForwardTests(unittest.TestCase):

    def test_grid_candidates_expands_product(self):
        self.assertEqual(len(_grid_candidates({'a': [1, 2], 'b': ['x', 'y']})), 4)

    def test_empty_windows_are_excluded_not_counted_as_zero(self):
        # Four signal days, no ACTIVE signals at all: nothing is rankable.
        days = [_day(f'2026-01-0{i + 1}', SESSIONS[i], AAA='SKIP') for i in range(4)]
        bars = {'AAA': _bars(*[FLAT] * 5)}
        grid = {'stop_loss_pct': [0.95], 'max_hold_days': [15], 'min_upside_pct': [1.5]}
        wf = walk_forward_optimize(days, bars, SESSIONS, train_window=2, test_window=1, grid=grid)
        self.assertEqual(wf['summary']['windows'], 2)
        self.assertEqual(wf['summary']['validation_windows'], 0)
        self.assertIsNone(wf['windows'][0]['selected_params'])
        self.assertIsNone(wf['latest_recommendation']['selected_params'])

    def test_training_is_cut_off_before_the_test_window(self):
        # A day-0 buy whose target only fills during the test window must be
        # 'open' (excluded) in training, not a training win.
        days = [_day('2026-01-02', SESSIONS[0], AAA='ACTIVE'), _day('2026-01-05', SESSIONS[1], BBB='SKIP'),
                _day('2026-01-06', SESSIONS[2], BBB='SKIP')]
        bars = {'AAA': _bars(FLAT, FLAT, (104, 106, 103, 105), FLAT), 'BBB': _bars(*[FLAT] * 4)}
        grid = {'stop_loss_pct': [0.95], 'max_hold_days': [15], 'min_upside_pct': [1.5]}
        wf = walk_forward_optimize(days, bars, SESSIONS, train_window=2, test_window=1, grid=grid,
                                   min_train_trades=1)
        self.assertEqual(wf['windows'][0]['train_stats']['n'], 0)


class StatsTests(unittest.TestCase):

    def test_empty_returns_zeroed(self):
        s = _stats([])
        self.assertEqual(s['n'], 0)
        self.assertFalse(s['significant'])
        self.assertFalse(s['sample_adequate'])
        self.assertIsNone(s['t_stat'])

    def test_single_value_no_std(self):
        s = _stats([1.5])
        self.assertEqual(s['n'], 1)
        self.assertAlmostEqual(s['mean'], 1.5)
        self.assertEqual(s['std'], 0.0)
        self.assertIsNone(s['t_stat'])  # cannot compute with n<2
        self.assertFalse(s['significant'])

    def test_sample_adequacy_threshold(self):
        self.assertFalse(_stats([0.1] * (MIN_ADEQUATE_SAMPLE - 1))['sample_adequate'])
        self.assertTrue(_stats([0.1] * MIN_ADEQUATE_SAMPLE)['sample_adequate'])

    def test_significant_requires_adequate_sample(self):
        # A tiny but perfectly consistent sample must NOT be called significant.
        small = _stats([1.0, 1.0, 1.0])
        self.assertFalse(small['significant'])
        self.assertFalse(small['sample_adequate'])

    def test_strong_consistent_signal_is_significant(self):
        # 40 trades, all clearly positive with low variance -> significant.
        rets = [1.0 + (0.01 if i % 2 else -0.01) for i in range(40)]
        s = _stats(rets)
        self.assertTrue(s['sample_adequate'])
        self.assertTrue(s['significant'])
        self.assertGreater(s['t_stat'], 1.96)

    def test_noisy_zero_mean_not_significant(self):
        # 40 trades centered on zero with high variance -> not significant.
        rets = [(-3.0 if i % 2 else 3.0) for i in range(40)]
        s = _stats(rets)
        self.assertTrue(s['sample_adequate'])
        self.assertFalse(s['significant'])

    def test_ci_brackets_mean(self):
        s = _stats([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertLessEqual(s['ci95_low'], s['mean'])
        self.assertGreaterEqual(s['ci95_high'], s['mean'])



class AttributionTests(unittest.TestCase):

    def test_summarize_trade_attribution_groups_models_and_execution_quality(self):
        journal = [
            {
                'ticker': 'NVDA',
                'outcome': 'win',
                'pnl_pct': 2.0,
                'entry_price': 99.0,
                'exit_price': 106.0,
                'consensus_buy_high': 100.0,
                'consensus_sell_low': 105.0,
                'predicting_models': ['a', 'b'],
            },
            {
                'ticker': 'AAPL',
                'outcome': 'loss',
                'pnl_pct': -1.0,
                'entry_price': 50.0,
                'exit_price': 49.5,
                'consensus_buy_high': 50.0,
                'consensus_sell_low': 52.0,
                'predicting_models': ['b'],
            },
        ]
        report = summarize_trade_attribution(journal)
        self.assertEqual(report['trades'], 2)
        self.assertEqual(report['overall']['wins'], 1)
        self.assertEqual(report['by_model']['b']['trades'], 2)
        self.assertEqual(report['by_model']['a']['avg_pnl_pct'], 2.0)
        self.assertGreater(report['execution_quality']['avg_price_improvement_pct'], 0)
        self.assertEqual(report['execution_quality']['profit_target_count'], 1)
        self.assertEqual(report['execution_quality']['stop_loss_count'], 1)



if __name__ == '__main__':
    unittest.main()
