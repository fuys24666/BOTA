"""Prepare a byte-preserving code archive; never stage, commit, fetch or push."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROOT_FILES = {'.gitattributes', '.gitignore', 'README.md', 'CITATION.cff',
              'LICENSE', 'THIRD_PARTY.md', 'pyproject.toml', 'requirements.txt'}
ROOT_DIRS = {'.github', 'configs', 'scripts', 'src', 'tests', 'docs', 'data_preprocess'}
PLACEHOLDERS = {'data/README.md', 'checkpoints/README.md', 'pretrained_models/README.md'}
SUFFIXES = {'.py', '.ps1', '.md', '.json', '.yaml', '.yml', '.txt', '.toml', '.cff'}
MAX_BYTES = 5 * 1024 * 1024
SECRET_PATTERNS = {
    'private_key': rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----',
    'github_token': rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})',
    'aws_access_key': rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
    'openai_key': rb'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{40,}',
    'huggingface_token': rb'\bhf_[A-Za-z0-9]{30,}',
    'private_overleaf_share': rb'https?://(?:www\.)?overleaf\.com/[0-9]+[a-z]+#[a-z0-9]+',
}


def git(*args: str, data: bytes | None = None) -> bytes:
    return subprocess.run(['git', *args], cwd=ROOT, input=data,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          check=True).stdout


def allowed(name: str) -> bool:
    p = Path(name)
    if p.is_absolute() or '..' in p.parts:
        return False
    if name.startswith('docs/audits/stage17_final_20261011/'):
        return False
    if any(x in p.parts for x in ('__pycache__', '.pytest_cache', '.git')):
        return False
    return (name in ROOT_FILES or name in PLACEHOLDERS or
            (p.parts[0] in ROOT_DIRS and p.suffix.lower() in SUFFIXES))


def inventory() -> list[str]:
    raw = git('ls-files', '-z', '--cached', '--others', '--exclude-standard')
    names = sorted(set(x.decode('utf-8') for x in raw.split(b'\0') if x))
    rejected = [n for n in names if not allowed(n)]
    if rejected:
        raise ValueError(f'Unexpected source files; review before packaging: {rejected}')
    return names


def inspect_bytes(name: str, data: bytes) -> None:
    if len(data) > MAX_BYTES:
        raise ValueError(f'File exceeds source-size limit: {name}')
    data.decode('utf-8-sig')  # This snapshot contains text source only.
    for label, pattern in SECRET_PATTERNS.items():
        if re.search(pattern, data):
            raise ValueError(f'Potential sensitive content ({label}): {name}; value suppressed')


def source_bytes(name: str) -> bytes:
    p = ROOT / name
    if p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError(f'Not a regular in-repository file: {name}')
    data = p.read_bytes()
    inspect_bytes(name, data)
    return data


def history_audit() -> dict:
    # Cached remote reference only; packaging must never access the network.
    git('rev-parse', '--verify', 'origin/main')
    objects = git('rev-list', '--objects', 'origin/main..HEAD').splitlines()
    blob_count = 0
    for record in objects:
        oid = record.split(b' ', 1)[0].decode('ascii')
        kind = git('cat-file', '-t', oid).strip()
        if kind != b'blob':
            continue
        name = record.split(b' ', 1)[1].decode('utf-8') if b' ' in record else oid
        if not allowed(name):
            raise ValueError(f'Unexpected file in outgoing history: {name}')
        size = int(git('cat-file', '-s', oid))
        if size > MAX_BYTES:
            raise ValueError(f'Oversized blob in outgoing history: {name}')
        inspect_bytes(name, git('cat-file', 'blob', oid))
        blob_count += 1
    return {'basis': 'cached origin/main (not fetched)', 'blobs_checked': blob_count,
            'ahead_commits': int(git('rev-list', '--count', 'origin/main..HEAD')),
            'behind_commits': int(git('rev-list', '--count', 'HEAD..origin/main'))}


def verify_index(contents: dict[str, bytes]) -> None:
    entries = {}
    for row in git('ls-files', '--stage', '-z').split(b'\0'):
        if not row:
            continue
        meta, name = row.split(b'\t', 1)
        mode, oid, stage = meta.split()
        if stage != b'0' or mode not in (b'100644', b'100755'):
            raise ValueError('Unexpected index mode or unresolved conflict')
        entries[name.decode('utf-8')] = oid.decode('ascii')
    if set(entries) != set(contents):
        raise ValueError('Git index file list differs from reviewed snapshot')
    algorithm = git('rev-parse', '--show-object-format').decode().strip()
    for name, data in contents.items():
        oid = hashlib.new(algorithm, b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if entries[name] != oid:
            raise ValueError(f'Git index bytes differ from snapshot: {name}; restage with --renormalize')
    print('VERIFIED: Git index preserves every snapshot file byte-for-byte.')


def verify(manifest: Path, check_index: bool = False) -> None:
    saved = json.loads(manifest.read_text(encoding='utf-8'))
    files = saved['files']
    if inventory() != sorted(files):
        raise ValueError('Source file list changed since packaging; prepare a new snapshot.')
    contents = {}
    for name, info in files.items():
        data = source_bytes(name)
        if hashlib.sha256(data).hexdigest() != info['sha256']:
            raise ValueError(f'File changed since packaging: {name}')
        contents[name] = data
    if check_index:
        verify_index(contents)
    print(f'VERIFIED: {len(files)} source files match the snapshot.')


def package(output: Path) -> None:
    output = output.resolve()
    if output.is_relative_to(ROOT.resolve()):
        raise ValueError('Output directory must be outside the repository.')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new or empty output directory; existing exports are preserved.')
    history = history_audit()
    contents = {name: source_bytes(name) for name in inventory()}
    output.mkdir(parents=True, exist_ok=True)
    archive = output / 'BOTA_source_20261017_stage27.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, data in contents.items():
            z.writestr('BOTA/' + name, data)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError('ZIP integrity check failed')
        for name, data in contents.items():
            if z.read('BOTA/' + name) != data:
                raise ValueError(f'ZIP byte mismatch: {name}')
    manifest = {
        'snapshot': '2026-10-17 / Stage 27',
        'head_before_upload': git('rev-parse', 'HEAD').decode().strip(),
        'history_audit': history,
        'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
        'files': {n: {'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()}
                  for n, b in contents.items()},
        'limitations': ['No network visibility check', 'Pattern scan is not a complete privacy audit',
                        'External experiment assets and model weights are not bundled'],
    }
    (output / 'source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    (output / 'git-paths.txt').write_text('\n'.join(contents) + '\n', encoding='utf-8')
    guide = f'''# Local GitHub upload — prepared source snapshot

Nothing has been staged, committed or pushed. Review README and source_manifest.json.
Keep the repository private unless you intentionally decide otherwise.
Run these blocks one at a time in PowerShell; stop on any unexpected result.

```powershell
Set-Location '{ROOT}'
python scripts/release/prepare_source_snapshot.py --verify '{output / 'source_manifest.json'}'
if ($LASTEXITCODE -ne 0) {{ throw 'Snapshot changed: regenerate before upload.' }}
git status --short
git diff --stat
```

Stage only the reviewed list (stop if the index already has unrelated work):

```powershell
if (git diff --cached --name-only) {{ throw 'Review the existing staged changes first.' }}
git --literal-pathspecs add --pathspec-from-file='{output / 'git-paths.txt'}'
if ($LASTEXITCODE -ne 0) {{ throw 'Staging failed.' }}
git --literal-pathspecs add --renormalize --pathspec-from-file='{output / 'git-paths.txt'}'
if ($LASTEXITCODE -ne 0) {{ throw 'Byte-preserving restaging failed.' }}
python scripts/release/prepare_source_snapshot.py --verify '{output / 'source_manifest.json'}' --verify-index
if ($LASTEXITCODE -ne 0) {{ throw 'Staged files do not match snapshot.' }}
git diff --cached --stat
git diff --cached --ignore-space-at-eol
```

After reviewing the staged changes:

```powershell
git commit -m "Update BOTA code and evaluation protocols through Stage 27"
if ($LASTEXITCODE -ne 0) {{ throw 'Commit failed; do not continue.' }}
git fetch origin
if ($LASTEXITCODE -ne 0) {{ throw 'Fetch failed; check authentication/network.' }}
git status -sb
git log --oneline --left-right HEAD...origin/main
```

Only push if on main with the expected remote and no remote-only commits:

```powershell
if ((git branch --show-current) -ne 'main') {{ throw 'Unexpected branch.' }}
if ((git remote get-url origin) -ne 'https://github.com/fuys24666/BOTA.git') {{ throw 'Unexpected remote.' }}
$BehindCount = git rev-list --count HEAD..origin/main
if ($LASTEXITCODE -ne 0) {{ throw 'Cannot compare with remote.' }}
if ([int]$BehindCount -ne 0) {{ throw 'Remote has new commits: reconcile before pushing; do not force push.' }}
git push origin main
if ($LASTEXITCODE -ne 0) {{ throw 'Push failed; do not force push.' }}
git status -sb
```

The cached remote comparison at preparation time showed {history['ahead_commits']} local
commits ahead and {history['behind_commits']} behind. The push includes those commits,
not just this new snapshot commit. Fetch above refreshes the comparison.
Do not run `git init`, delete `.git`, or upload weights/data to fix a push failure.
The ZIP is a source backup, not a replacement for the existing Git history.
'''
    (output / 'UPLOAD_INSTRUCTIONS.md').write_text(guide, encoding='utf-8')
    print(json.dumps({'archive': str(archive), 'files': len(contents),
                      'zip_bytes': archive.stat().st_size, 'history_audit': history}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--output-dir', type=Path)
    group.add_argument('--verify', type=Path)
    parser.add_argument('--verify-index', action='store_true', help='Also compare the staged bytes with --verify')
    args = parser.parse_args()
    if args.verify_index and not args.verify:
        parser.error('--verify-index requires --verify')
    verify(args.verify, args.verify_index) if args.verify else package(args.output_dir)
