"""Additive, audited loader repair; the original Stage25 contract stays immutable."""
import argparse
from pathlib import Path

from . import ali_stage25_historical_mask_audit_v1 as s

ORIGINAL_DATASET = s.dataset


def dataset(workspace, cohort, confirm):
    s.authorize(cohort['scope'], confirm)
    if cohort['scope'] != 'FinalTest' or cohort['lane'] != 'binary':
        return ORIGINAL_DATASET(workspace, cohort, confirm)
    path = Path(cohort['data_path'])
    digest = cohort.get('data_sha256')
    if not digest or s.sha256_file(path) != digest:
        raise ValueError('FinalTest content changed or missing frozen hash')
    base = s.s24.s23.runner._base_t5_config(workspace, cohort['benchmark'])
    model_dir = Path(base['paths']['model_dir'])
    if not model_dir.is_absolute():
        model_dir = workspace / model_dir
    tok = s.T5Tokenizer.from_pretrained(str(model_dir), local_files_only=True)
    # This subclass retains the identical item/collation implementation, but
    # requires the existing Stage16 explicit FinalTest capability at construction.
    data = s.inventory.s16.BinaryDataset(path, tok, s.inventory.s16.CAPABILITY)
    if not cohort['indices'] or min(cohort['indices']) < 0 or max(cohort['indices']) >= len(data):
        raise ValueError('cohort indices out of range')
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--workspace', type=Path, default=Path('C:/python/E2URec-main'))
    parser.add_argument('--mode', choices=['Preflight', 'Full', 'Analyze'], default='Preflight')
    parser.add_argument('--scope', choices=['Development', 'FinalTest'], default='Development')
    parser.add_argument('--lane', choices=['All', 'binary', 'ranking'], default='All')
    parser.add_argument('--confirm-finaltest', action='store_true')
    args = parser.parse_args()
    if args.mode == 'Full':
        s.authorize(args.scope, args.confirm_finaltest)
    repo, workspace = args.repo.resolve(), args.workspace.resolve()
    root = workspace / s.ROOT
    protocol = s.verify(root / 'protocol')
    s.check_plan(protocol['plan'], s.load_config(repo))
    if args.mode == 'Full':
        development = s.verify(root / 'summary/Development', protocol['contract'])
        if development['status'] not in ('COMPLETED', 'COMPLETED_WITH_UNAVAILABLE'):
            raise RuntimeError('Development completion required')
        repair = {
            'status': 'LOADER_REPAIR01',
            'repair_sha256': s.sha256_file(Path(__file__)),
            'launcher_sha256': s.sha256_file(repo / 'scripts/bota_if/run_ali_stage25_loader_repair01.ps1'),
            'parent_contract': protocol['contract'],
            'change': 'Use capability-gated Stage16 BinaryDataset for authorized FinalTest binary only',
            'selection_performed': False,
        }
        with s.s24.s23.s20.lock(root):
            receipt = s.task(root, 'repairs/finaltest_loader01', protocol['contract'], lambda stage: repair)
            if any(receipt.get(k) != v for k, v in repair.items()):
                raise ValueError('repair implementation changed after admission')
    s.dataset = dataset
    try:
        s.main()
    finally:
        s.dataset = ORIGINAL_DATASET


if __name__ == '__main__':
    main()
