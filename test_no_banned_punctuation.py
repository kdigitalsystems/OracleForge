"""Guard: no AI-looking punctuation in user-facing site content.

The dashboard published to GitHub Pages must not contain em dashes,
en dashes, or curly quotes (a hard requirement across this user's web
projects; the site was swept clean on 2026-08-25). This test keeps the
characters from creeping back in two places:

- string literals in scripts/generate_html_report.py (every user-visible
  string on the site originates here), extracted with tokenize so code
  comments and docstrings, which never reach the page, cannot false-positive
- the committed docs/index.html output itself, scanned whole

Kept as unittest so the nightly forge's existing test step picks it up with
no extra CI wiring.
"""
from __future__ import annotations

import io
import os
import token
import tokenize
import unittest

BANNED = {
    '—': 'em dash',
    '–': 'en dash',
    '‘': 'left single curly quote',
    '’': 'right single curly quote',
    '“': 'left double curly quote',
    '”': 'right double curly quote',
}

GENERATOR = os.path.join('scripts', 'generate_html_report.py')
SITE_OUTPUT = os.path.join('docs', 'index.html')


def _violations_in_string_literals(path: str) -> list[str]:
    out = []
    with open(path, encoding='utf-8') as f:
        source = f.read()
    prev_significant = None
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == token.STRING:
            # Skip docstrings: a string opening a logical line right after a
            # NEWLINE/INDENT/DEDENT (or at file start) is documentation, not
            # page content.
            if prev_significant in (None, token.NEWLINE, token.INDENT, token.DEDENT):
                prev_significant = tok.type
                continue
            for ch, name in BANNED.items():
                if ch in tok.string:
                    out.append(f'{path}:{tok.start[0]}: {name} in string literal')
        if tok.type not in (token.NL, token.COMMENT):
            prev_significant = tok.type
    return out


class BannedPunctuationTests(unittest.TestCase):

    def test_generator_string_literals_are_clean(self):
        violations = _violations_in_string_literals(GENERATOR)
        self.assertEqual(violations, [], '\n'.join(violations))

    def test_generated_site_is_clean(self):
        if not os.path.exists(SITE_OUTPUT):
            self.skipTest('docs/index.html not present in this checkout')
        with open(SITE_OUTPUT, encoding='utf-8') as f:
            html = f.read()
        found = [
            f'{name} (x{html.count(ch)})'
            for ch, name in BANNED.items() if ch in html
        ]
        self.assertEqual(found, [], f'banned characters in {SITE_OUTPUT}: ' + ', '.join(found))


if __name__ == '__main__':
    unittest.main()
