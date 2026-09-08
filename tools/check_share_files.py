"""Check this small maintenance bundle, offline, without reading clinical files.

This is a limited static check, not a guarantee that arbitrary files are safe to share.
Keep SHARE_FILES explicit; new deliverables require a deliberate list update.
"""
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SHARE_FILES = (
    'README.md', 'AGENTS.md', 'agent.md', 'HANDOFF.md', 'CONTRIBUTING.md',
    '.gitignore', '.gitattributes',
    'docs/PIPELINE.md', 'docs/EVIDENCE.md', 'docs/DECISIONS.md', 'docs/RESULTS.md',
    'templates/RESULT.md', '.github/ISSUE_TEMPLATE/task.md',
    '.github/ISSUE_TEMPLATE/result.md', '.github/pull_request_template.md',
    'tools/check_share_files.py',
)


def validate(root=ROOT):
    root = Path(root).resolve()
    errors, total = [], 0
    for name in SHARE_FILES:
        path = root / name
        if path.is_symlink() or not path.is_file():
            errors.append(f'{name}: missing or symbolic link')
            continue
        if not path.resolve().is_relative_to(root):
            errors.append(f'{name}: outside repository')
            continue
        data = path.read_bytes()
        total += len(data)
        if len(data) > 512_000:
            errors.append(f'{name}: unexpectedly large for a maintenance file')
        try:
            text = data.decode('utf-8')
        except UnicodeDecodeError:
            errors.append(f'{name}: not UTF-8')
            continue
        if '\ufffd' in text or '\x00' in text:
            errors.append(f'{name}: invalid text characters')
        for label, pattern in (
            ('private download link', r'https?://(?:www\.)?dropbox\.com/'),
            ('access token', r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{24,})'),
            ('personal Windows path', r'[A-Za-z]:[\\/]Users[\\/][^\s]+'),
            ('private key', r'-----BEGIN [A-Z ]*PRIVATE KEY-----'),
        ):
            if re.search(pattern, text):
                errors.append(f'{name}: possible {label}')
        if path.suffix != '.md':
            continue
        # Exclude fenced code before examining Markdown links.
        prose = re.sub(r'```.*?```', '', text, flags=re.S)
        for href in re.findall(r'\[[^\]\n]+\]\(([^)\n]+)\)', prose):
            href = href.strip('<>')
            parsed = urlsplit(href)
            if parsed.scheme or href.startswith('#'):
                continue
            target = (path.parent / unquote(parsed.path)).resolve()
            if not target.is_relative_to(root) or not target.is_file():
                errors.append(f'{name}: missing local link {href}')
            elif target.relative_to(root).as_posix() not in SHARE_FILES:
                errors.append(f'{name}: link points outside the share bundle: {href}')
    return errors, total


def tracked_extras():
    # Do not inspect the parent Desktop repository when this directory is not a repo.
    if not (ROOT / '.git').exists():
        return []
    result = subprocess.run(
        ['git', '-C', str(ROOT), 'ls-files', '-z'],
        capture_output=True, check=False,
    )
    if result.returncode:
        return ['could not inspect tracked files']
    tracked = set(result.stdout.decode('utf-8').split('\0')) - {''}
    return [f'unreviewed tracked file: {name}' for name in sorted(tracked - set(SHARE_FILES))]


def main():
    errors, total = validate()
    errors.extend(tracked_extras())
    if errors:
        for error in errors:
            print('FAIL:', error)
        return 1
    print(f'PASS: {len(SHARE_FILES)} maintenance files, {total:,} bytes; UTF-8 and local links checked.')
    print('Scope: static bundle check only; no network, patient data or experiment execution.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
