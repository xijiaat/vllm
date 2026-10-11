"""Report all fixed crossover phases; no selection of favorable samples."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text())


def describe(values):
    return {'all': values, 'median': statistics.median(values),
            'min': min(values), 'max': max(values),
            'cv_pct': statistics.stdev(values) / statistics.mean(values) * 100}


def main(root):
    metadata = read(root / 'plan.json')
    plan = metadata['plan']
    assert read(root / 'complete.json')['complete']
    result = {'complete': True, 'automatic_stable_gain_claim': False,
              'plan': metadata, 'cells': [], 'model_inventory': []}
    imports = []
    inventories = {}
    for run in sorted((root / 'runs').iterdir()):
        owner = read(run / 'raw/owner.json')
        if owner['arm'] is None:
            continue
        assert read(run / 'raw/stopped.json')['state']['Running'] is False
        source = read(run / 'raw/import-proof.json')
        imports.append(source)
        ranks = [read(p) for p in sorted((run / 'raw/inventory').glob('rank-*.json'))]
        assert len(ranks) == 4
        assert [x['gpu_uuid'] for x in ranks] == [g[4:] if g.startswith('GPU-') else g for g in owner['gpus']]
        for rank in ranks:
            assert rank['arm'] == owner['arm'] and rank['synthetic_acceptance'] is False
            compressors = rank['models']['target']['compressors']
            assert len(compressors) == 4
            assert all(c['fusion'] == (owner['arm'] == 'B') for c in compressors)
        inventories[(owner['phase'], owner['arm'])] = ranks
        result['model_inventory'].append({'phase': owner['phase'], 'arm': owner['arm'],
            'group': owner['group'], 'gpu_order': owner['gpus'], 'ranks': ranks,
            'source_proof': source, 'runid': owner['runid']})
    assert len(imports) == 4
    assert all(p['tilelang_seed_files'] == imports[0]['tilelang_seed_files'] for p in imports)
    for phase in (0, 1):
        assert read(root / f'phase{phase}/bench/complete.json')['complete']
        for a, b in zip(inventories[(phase, 'A')], inventories[(phase, 'B')]):
            for role in ('target', 'draft'):
                assert a['models'][role]['linear_kernels'] == b['models'][role]['linear_kernels']
                assert a['models'][role]['moe_methods'] == b['models'][role]['moe_methods']
    for mode, length, concurrency in plan['cells']:
        name = f'{mode}-{length}-c{concurrency}'
        entry = {'cell': name, 'phases': [], 'pooled': {}}
        all_arm = {'A': [], 'B': []}
        for phase in (0, 1):
            cell = root / f'phase{phase}/bench' / name
            phase_result = {'phase': phase, 'placement': plan['phases'][phase]}
            rows = {}
            for arm in ('A', 'B'):
                paths = sorted(cell.glob(f'sample-*-{arm}/sample.json'))
                assert len(paths) == 3, (cell, arm, len(paths))
                rows[arm] = [read(path) for path in paths]
                for pair, row in enumerate(rows[arm]):
                    assert row['arm'] == arm and row['phase'] == phase and row['pair'] == pair
                    assert row['input_sha256'] == metadata['prompts_sha256']
                    assert row['length'] == length and row['concurrency'] == concurrency
                    assert row['output_tokens_per_request'] == 1024
                    assert row['metrics_delta']['num_preemptions_total'] == 0
                    assert row['metrics_delta']['dspark']['num_drafts'] > 0
                    assert len(row['requests']) == 2 * concurrency
                    assert all(r['usage']['completion_tokens'] == 1024 for r in row['requests'])
                    if mode == 'cold':
                        assert row['cache_hit_tokens'] == 0
                    peer = read(paths[pair].parent / 'peer-isolation.json')
                    assert peer['arm'] == arm and peer['phase'] == phase
                all_arm[arm].extend(rows[arm])
                phase_result[arm] = {
                    'output_tok_s': describe([r['output_tok_s'] for r in rows[arm]]),
                    'ttft_s': describe([r['metrics_delta']['time_to_first_token_seconds']['mean'] for r in rows[arm]]),
                    'tpot_s': describe([r['metrics_delta']['request_time_per_output_token_seconds']['mean'] for r in rows[arm]]),
                    'e2e_s': describe([r['metrics_delta']['e2e_request_latency_seconds']['mean'] for r in rows[arm]]),
                    'acceptance_length': describe([r['metrics_delta']['dspark']['mean_acceptance_length'] for r in rows[arm]]),
                    'cache_hit_ratios': [r['cache_hit_ratio'] for r in rows[arm]],
                }
            a, b = (phase_result[arm]['output_tok_s']['median'] for arm in ('A', 'B'))
            phase_result['change_pct'] = (b / a - 1) * 100
            phase_result['paired_change_pct'] = [(b['output_tok_s'] / a['output_tok_s'] - 1) * 100
                                                for a, b in zip(rows['A'], rows['B'])]
            phase_result['paired_output_hash_matches'] = [sum(a['output_sha256'] == b['output_sha256']
                for a, b in zip(ar['requests'], br['requests'])) for ar, br in zip(rows['A'], rows['B'])]
            assert phase_result['A']['cache_hit_ratios'] == phase_result['B']['cache_hit_ratios']
            entry['phases'].append(phase_result)
        for arm in ('A', 'B'):
            entry['pooled'][arm] = describe([r['output_tok_s'] for r in all_arm[arm]])
        ratios = [1 + p['change_pct'] / 100 for p in entry['phases']]
        entry['crossover_geometric_change_pct'] = (math.sqrt(ratios[0] * ratios[1]) - 1) * 100
        entry['pooled_median_change_pct'] = (entry['pooled']['B']['median'] / entry['pooled']['A']['median'] - 1) * 100
        result['cells'].append(entry)
        print(name, 'phase_pct', [round(p['change_pct'], 4) for p in entry['phases']],
              'crossover_pct', round(entry['crossover_geometric_change_pct'], 4),
              'pooled_medians', [round(entry['pooled'][a]['median'], 3) for a in ('A', 'B')])
    out = root / 'analysis.json'
    out.write_text(json.dumps(result, indent=2) + '\n')
    print('ANALYSIS', str(out), hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    main(p.parse_args().root)
