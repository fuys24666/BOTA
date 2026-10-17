"""Strict additive audit for Stage21's residual-rejection reporting repair."""
import copy
from pathlib import Path
from . import ali_stage20_amazon_baselines_v1 as parent
from . import ali_stage21_amazon_seed43_kernel_v1 as k

RECORD='configs/ali_stage21_repair01_v1.json'
ALLOWED={'src/bota_short_benchmark/ali_stage21_amazon_seed43_v1.py',
         'src/bota_short_benchmark/ali_stage21_amazon_seed43_kernel_v1.py'}

def check(repo,workspace,p,root):
    r=k.read(repo/RECORD)
    if set(r['migrations'])!=ALLOWED or r['id']!='stage21-repair01':raise ValueError('unexpected repair scope')
    checks=copy.deepcopy(p['checks'])
    for name,bindings in r['migrations'].items():
        if checks['source_sha256'].get(name) not in (bindings['before'],bindings['after']):
            raise ValueError(f'unrecognized source predecessor: {name}')
        if k.sha256_file(repo/name)!=bindings['after']:raise ValueError(f'repair source differs: {name}')
        checks['source_sha256'][name]=bindings['after']
    if k.sha256_file(Path(__file__))!=r['repair_validator_sha256']:raise ValueError('repair validator changed')
    parent.s19.check_inputs(repo,checks)
    # Append provenance. Never rewrite protocol/result.json or its manifest.
    record={'id':r['id'],'repair_record_sha256':k.sha256_file(repo/RECORD),
            'parent_contract':k.canonical_hash(p),'repair':r,'test_accessed':False}
    k.freeze(root,'implementation_repairs/repair01',record)
    return record
