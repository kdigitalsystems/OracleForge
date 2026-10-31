"""Tests for scripts/runner_cleanup.sh, run against decoy directories."""
import os
import subprocess
import tempfile
import time
import unittest

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts', 'runner_cleanup.sh')


def _write(path, text='x'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(text)


def _git(cwd, *args):
    subprocess.run(['git', '-C', cwd, '-c', 'user.email=t@t', '-c', 'user.name=t', *args],
                   check=True, capture_output=True)


class RunnerCleanupTests(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = self._tmp.name
        self.ws = os.path.join(root, 'work', 'OracleForge', 'OracleForge')
        self.tmpd = os.path.join(root, 'tmp')
        self.pip = os.path.join(root, 'home', '.cache', 'pip')
        self.other = os.path.join(root, 'work', 'OtherProject')   # a neighbour's workspace
        os.makedirs(self.ws)
        _git(self.ws, 'init', '-q')
        _write(os.path.join(self.ws, '.gitignore'), '__pycache__/\n*.tmp1\n')
        _write(os.path.join(self.ws, 'tracked.py'), 'original\n')
        _git(self.ws, 'add', '-A')
        _git(self.ws, 'commit', '-q', '-m', 'init')
        os.makedirs(self.tmpd)
        _write(os.path.join(self.other, 'precious.txt'))

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, workspace=None, **env):
        e = {**os.environ, 'GITHUB_WORKSPACE': workspace or self.ws, 'CLEANUP_TMP': self.tmpd,
             'CLEANUP_PIP_CACHE': self.pip, 'HOME': os.path.join(self._tmp.name, 'home'), **env}
        return subprocess.run(['bash', SCRIPT], capture_output=True, text=True, env=e)

    def test_removes_untracked_ignored_and_pycache_but_keeps_tracked_work(self):
        _write(os.path.join(self.ws, 'untracked.log'))
        _write(os.path.join(self.ws, 'state.json.tmp1'))                 # ignored leftover
        _write(os.path.join(self.ws, 'pkg', '__pycache__', 'm.pyc'))
        _write(os.path.join(self.ws, 'tracked.py'), 'edited, not committed\n')
        r = self._run()
        self.assertEqual(r.returncode, 0, r.stderr)
        for gone in ('untracked.log', 'state.json.tmp1', os.path.join('pkg', '__pycache__')):
            self.assertFalse(os.path.exists(os.path.join(self.ws, gone)), gone)
        self.assertEqual(open(os.path.join(self.ws, 'tracked.py')).read(), 'edited, not committed\n')
        self.assertTrue(os.path.isdir(os.path.join(self.ws, '.git')))

    def test_removes_our_temp_files_and_leaves_other_temp_files(self):
        for name in ('ollama.log', 'ollama_tags.json', 'ollama_canary.log'):
            _write(os.path.join(self.tmpd, name))
        _write(os.path.join(self.tmpd, 'oracleforge-canary-abc', 'history', 'p.json'))
        _write(os.path.join(self.tmpd, 'someone_elses.log'))
        self._run()
        self.assertEqual(sorted(os.listdir(self.tmpd)), ['someone_elses.log'])

    def test_pip_cache_only_loses_entries_not_read_for_30_days(self):
        old, new = os.path.join(self.pip, 'http', 'old_wheel'), os.path.join(self.pip, 'http', 'new_wheel')
        _write(old); _write(new); _write(os.path.join(self.pip, 'wheels', 'stale', 'w'))
        past = time.time() - 40 * 86400
        for p in (old, os.path.join(self.pip, 'wheels', 'stale', 'w')):
            os.utime(p, (past, past))
        self._run()
        self.assertFalse(os.path.exists(old))
        self.assertFalse(os.path.exists(os.path.join(self.pip, 'wheels', 'stale')), 'emptied dir removed')
        self.assertTrue(os.path.exists(new), 'recently used entries stay')

    def test_never_touches_a_neighbours_workspace(self):
        self._run()
        self.assertTrue(os.path.exists(os.path.join(self.other, 'precious.txt')))

    def test_refuses_to_clean_the_users_home(self):
        # Only decoy paths are ever passed in: an unguarded run against a real
        # path such as / would delete files on this machine if the guard broke.
        home = os.path.join(self._tmp.name, 'home')
        _write(os.path.join(home, 'keep_me.txt'))
        _write(os.path.join(home, 'project', '__pycache__', 'm.pyc'))
        r = self._run(workspace=home)
        self.assertIn('refusing', r.stdout)
        self.assertTrue(os.path.exists(os.path.join(home, 'keep_me.txt')))
        self.assertTrue(os.path.exists(os.path.join(home, 'project', '__pycache__', 'm.pyc')))

    def test_missing_workspace_is_a_clean_no_op(self):
        r = self._run(workspace=os.path.join(self._tmp.name, 'does_not_exist'))
        self.assertEqual(r.returncode, 0)
        self.assertIn('nothing to do', r.stdout)

    def test_idempotent_and_reports_free_space(self):
        first = self._run(); second = self._run()
        self.assertEqual((first.returncode, second.returncode), (0, 0))
        self.assertRegex(second.stdout, r'cleanup: \d+ GiB free on .*; this step freed -?\d+ MiB')


if __name__ == '__main__':
    unittest.main()
