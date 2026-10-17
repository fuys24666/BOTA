from __future__ import annotations

import importlib.util
import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('source_snapshot', ROOT / 'scripts/release/prepare_source_snapshot.py')
SNAPSHOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SNAPSHOT)


@pytest.mark.parametrize('name', [
    'src/bota_short_benchmark/ali_stage27_amazon_receraser_rescore_v1.py',
    'docs/audits/ali_stage16_result_audit_v1.json', '.gitattributes', 'data/README.md',
])
def test_release_keeps_required_code_and_audits(name):
    assert SNAPSHOT.allowed(name)


@pytest.mark.parametrize('name', [
    '.env', 'data/users.json', 'outputs/metrics.json', 'checkpoints/model.pt',
    'docs/audits/stage17_final_20261011/statistics.json', 'docs/review.png',
    '../private.py', 'src/__pycache__/module.pyc',
])
def test_release_excludes_local_artifacts(name):
    assert not SNAPSHOT.allowed(name)


def test_release_rejects_sensitive_content_without_echoing_it():
    fake = b'ghp_' + b'x' * 36
    with pytest.raises(ValueError, match='value suppressed') as exc:
        SNAPSHOT.inspect_bytes('example.txt', fake)
    assert fake.decode() not in str(exc.value)


def test_frozen_source_bytes_are_not_normalized():
    assert '* -text' in (ROOT / '.gitattributes').read_text(encoding='utf-8')


@pytest.mark.parametrize('index_data,ok', [(b'a\r\n', True), (b'a\n', False)])
def test_index_check_detects_newline_conversion(monkeypatch, index_data, ok):
    oid = hashlib.sha1(b'blob ' + str(len(index_data)).encode() + b'\0' + index_data).hexdigest()
    def fake_git(*args):
        if args[0] == 'ls-files':
            return f'100644 {oid} 0\tREADME.md\0'.encode()
        return b'sha1\n'
    monkeypatch.setattr(SNAPSHOT, 'git', fake_git)
    if ok:
        SNAPSHOT.verify_index({'README.md': b'a\r\n'})
    else:
        with pytest.raises(ValueError, match='bytes differ'):
            SNAPSHOT.verify_index({'README.md': b'a\r\n'})
