"""Read-only-source diagnostic of Amazon legacy BOTA's canonical trajectory.

No Development scoring or lambda selection. Separate diagnostic output only.
"""
from pathlib import Path
import os
import time

import torch

from . import ali_stage18_amazon_lambda_search_v1 as s
from src.paper_if_a2.artifacts import atomic_torch_save


def main():
    repo = Path(__file__).resolve().parents[2]
    workspace = Path('C:/python/E2URec-main')
    if (workspace / 'outputs/ali_stage18_reproduction_diagnostic/report.json').exists():
        raise FileExistsError('Preserving completed legacy diagnostic; do not overwrite its report.')
    path = repo / 'configs/ali_stage18_amazon_lambda_search_v1.json'
    c = s.load_config(repo, path)
    checks, b, reg, _ = s.audit(repo, workspace, c, path)
    root = workspace / 'outputs/ali_stage18_reproduction_diagnostic'
    root.mkdir(parents=True, exist_ok=True)
    os.chdir(workspace)
    device = s.gpu()
    s.seed_everything(42)
    model, names, params, bases, _, tok, data, users = s.runner._model_context(workspace, b, reg, device)
    targets = sorted({u for r in reg['scenarios'] for u in r['users']})
    class ProgressBudget(s.s8.StepBudget):
        def step(self, optimizer, arm):
            super().step(optimizer, arm)
            if self.calls == 1 or self.calls % 10 == 0:
                print(f'[Legacy diagnostic] step={self.calls}/200', flush=True)
    begun = time.perf_counter()
    canonical, states, trace, optimizer_hash, rng_hash = s.s8.p2b.run_canonical_full(
        model, data, reg['order'], users, targets, params, names, device, tok.pad_token_id,
        s.runner._engine(workspace, b), ProgressBudget(200))
    original = torch.load(Path(checks['references']['Original-Short'])/'scenarios/K2/adapter/adapter_model.pt', map_location='cpu', weights_only=True)
    report = {'test_accessed': False, 'development_scored': False, 'steps': 200,
              'canonical_matches_original': all(torch.equal(canonical[n],original['B'][n]) for n in names),
              'scenarios': {}, 'seconds': time.perf_counter()-begun}
    flat = lambda state: torch.cat([state[n].float().flatten() for n in names])
    for r in reg['scenarios']:
        sid = r['id']
        old = torch.load(Path(checks['references']['BOTA-T2-Short'])/'scenarios'/sid/'adapter/adapter_model.pt', map_location='cpu', weights_only=True)
        slots = [targets.index(u) for u in r['users']]
        values = {n: canonical[n]+states['T2_AdamW_full_state'][i]['theta'][slots].sum(0).float() for i,n in enumerate(names)}
        update = flat(values)-flat(canonical)
        report['scenarios'][sid] = {
            'legacy_replay_matches_old_bota': all(torch.equal(values[n],old['B'][n]) for n in names),
            'legacy_update_norm': float(torch.linalg.vector_norm(update.double())),
            'canonical_gap_norm': float(torch.linalg.vector_norm((flat(canonical)-flat(original['B'])).double())),
            'old_minus_formal_original_norm': float(torch.linalg.vector_norm((flat(old['B'])-flat(original['B'])).double())),
            'legacy_replay_alignment': s.s8.p2b.vector_metrics(flat(old['B']),flat(values))}
        manifest = s.read(Path(checks['references']['BOTA-T2-Short'])/'scenarios'/sid/'scenario_manifest.json')
        report['scenarios'][sid]['old_optimizer_hash_matches'] = optimizer_hash == manifest['optimizer_state_sha256']
        report['scenarios'][sid]['old_rng_hash_matches'] = rng_hash == manifest['rng_sha256']
        report['scenarios'][sid]['old_trace_hash_matches'] = s.canonical_hash(trace) == manifest['canonical_trace_sha256']
    atomic_torch_save(root/'legacy_replay.pt', {'A': bases, 'B': canonical, 'states': states})
    s.atomic_json(root/'report.json', report)
    print(report, flush=True)


if __name__ == '__main__':
    main()
