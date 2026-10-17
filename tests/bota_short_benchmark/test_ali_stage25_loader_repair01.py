import json
from types import SimpleNamespace
import pytest
import torch
from src.bota_short_benchmark import ali_stage25_finaltest_loader_repair01 as r


class Tokenizer:
    pad_token_id = 0
    def encode(self, text, **kwargs):
        return [len(text), 1]


def fixture(tmp_path, monkeypatch):
    path = tmp_path / 'test' / 'test_10_simple.json'
    path.parent.mkdir()
    path.write_text(json.dumps([{'input': 'hello', 'output': 'yes'}]), encoding='utf-8')
    cohort = {'scope': 'FinalTest', 'lane': 'binary', 'data_path': str(path),
              'data_sha256': r.s.sha256_file(path), 'indices': [0], 'benchmark': {}}
    monkeypatch.setattr(r.s.s24.s23.runner, '_base_t5_config', lambda *a: {'paths': {'model_dir': 'model'}})
    monkeypatch.setattr(r.s, 'T5Tokenizer', SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer()))
    return path, cohort


def test_authorized_loader_preserves_samples_and_collation(tmp_path, monkeypatch):
    path, cohort = fixture(tmp_path, monkeypatch)
    data = r.dataset(tmp_path, cohort, True)
    assert len(data) == 1
    assert data[0]['input_ids'].tolist() == [5, 1]
    legacy_path = tmp_path / 'development.json'
    legacy_path.write_bytes(path.read_bytes())
    legacy = r.s.JsonPromptDataset(legacy_path, Tokenizer())
    a, b = data.collate_fn([data[0]]), legacy.collate_fn([legacy[0]])
    assert all(torch.equal(a[k], b[k]) for k in a)
    with pytest.raises(ValueError, match='test paths are forbidden'):
        r.s.JsonPromptDataset(path, Tokenizer())


def test_no_authorization_before_access(tmp_path, monkeypatch):
    monkeypatch.setattr(r.s, 'sha256_file', lambda *a: pytest.fail('must not read'))
    with pytest.raises(PermissionError):
        r.dataset(tmp_path, {'scope': 'FinalTest'}, False)


@pytest.mark.parametrize('digest', [None, 'wrong'])
def test_bad_or_missing_hash_rejected(tmp_path, monkeypatch, digest):
    _, cohort = fixture(tmp_path, monkeypatch)
    cohort['data_sha256'] = digest
    with pytest.raises(ValueError, match='hash'):
        r.dataset(tmp_path, cohort, True)


@pytest.mark.parametrize('scope,lane', [('Development', 'binary'), ('FinalTest', 'ranking')])
def test_other_paths_unchanged(tmp_path, monkeypatch, scope, lane):
    marker = object()
    monkeypatch.setattr(r, 'ORIGINAL_DATASET', lambda *a: marker)
    assert r.dataset(tmp_path, {'scope': scope, 'lane': lane}, True) is marker
