"""Alternating full HTTP requests against two already-ready, disjoint TP4 engines."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

import bench_helpers as h


def settle(client):
    for _ in range(150):
        raw = client.call('/metrics').decode()
        if h.is_idle(raw):
            time.sleep(0.3)
            final = client.call('/metrics').decode()
            if h.is_idle(final):
                return final
        time.sleep(0.2)
    raise AssertionError('engine failed to become idle')


def reset(client):
    settle(client)
    assert json.loads(client.call('/reset_prefix_cache', {}))['success'] is True


def prime(client, rows, out):
    out.mkdir()
    hits = []
    for pass_id in range(2):
        for index, row in enumerate(rows):
            response = json.loads(client.call('/v1/completions', {
                'model': 'dsv41', 'prompt': row, 'add_special_tokens': False,
                'max_tokens': 1, 'temperature': 0, 'top_p': 1,
                'seed': 713, 'ignore_eos': True,
            }))
            h.save(out / f'{pass_id}-{index:02d}.json', response)
            assert response['usage']['prompt_tokens'] == len(row)
            assert response['usage']['completion_tokens'] == 1
            if pass_id == 1:
                hits.append(response['usage']['prompt_tokens_details']['cached_tokens'])
    assert len(set(hits)) == 1 and 0 < hits[0] < len(rows[0]), hits
    settle(client)
    return hits[0]


def unchanged_peer(before, after):
    h.idle(before)
    h.idle(after)
    for key in ('request_success_total', 'prompt_tokens_total',
                'generation_tokens_total', 'num_preemptions_total',
                'spec_decode_num_drafts_total'):
        assert h.delta(before, after, 'vllm:' + key) == 0, ('peer interference', key)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', type=int, choices=[0, 1], required=True)
    args = p.parse_args()
    plan = json.loads(Path('/code/plan.json').read_text())
    root = Path('/campaign') / f'phase{args.phase}' / 'bench'
    root.mkdir(exist_ok=False)
    h.save(root / 'plan.json', plan)
    rows_path = Path('/campaign/shared/prompts.json')
    rows = json.loads(rows_path.read_text())
    assert len(rows) == 32 and all(len(row) == 16384 for row in rows)
    assert len({tuple(row[:64]) for row in rows}) == 32
    h.save(root / 'input-proof.json', {'sha256': h.sha(rows_path), 'sessions': 32})
    clients = {arm: h.Client('http://127.0.0.1:' + str(port), timeout=900)
               for arm, port in plan['ports'].items()}
    try:
        for mode, length, concurrency in plan['cells']:
            cell = root / f'{mode}-{length}-c{concurrency}'
            cell.mkdir()
            selected = [row[:length - 64] + row[-64:] for row in rows[:2 * concurrency]]
            expected = {'A': 0, 'B': 0}
            if mode == 'warm':
                for arm in ('A', 'B'):
                    peer = clients['B' if arm == 'A' else 'A']
                    before = settle(peer)
                    reset(clients[arm])
                    expected[arm] = prime(clients[arm], selected, cell / ('prime-' + arm))
                    unchanged_peer(before, settle(peer))
                assert expected['A'] == expected['B'], expected
            for kind, orders in (('calibration', plan['calibration_orders']),
                                 ('sample', plan['measurement_orders'])):
                for pair, original in enumerate(orders):
                    order = original[::-1] if args.phase == 1 else original
                    for arm in order:
                        client = clients[arm]
                        peer = clients['B' if arm == 'A' else 'A']
                        peer_before = settle(peer)
                        if mode == 'cold':
                            reset(client)
                        settle(client)
                        out = cell / f'{kind}-{pair:02d}-{arm}'
                        out.mkdir()
                        report = h.measure(client, selected, length, concurrency,
                                           'dsv41', True, out, expected[arm])
                        peer_after = settle(peer)
                        h.save(out / 'peer-isolation.json', {
                            'before': peer_before, 'after': peer_after,
                            'arm': arm, 'phase': args.phase,
                        })
                        unchanged_peer(peer_before, peer_after)
                        report.update(arm=arm, phase=args.phase, mode=mode,
                                      pair=pair, order=order, kind=kind,
                                      input_sha256=h.sha(rows_path),
                                      full_warm_means_engine_supported_limit_not_100_percent=False,
                                      cache_hit_source='observed priming' if mode == 'warm' else 'reset; exact zero checked per request')
                        (out / 'sample.json').write_text(json.dumps(report, indent=2))
                        print(json.dumps({
                            'event': 'SAMPLE', 'phase': args.phase, 'cell': cell.name,
                            'kind': kind, 'pair': pair, 'arm': arm,
                            'output_tok_s': report['output_tok_s'],
                            'ttft_s': report['metrics_delta']['time_to_first_token_seconds']['mean'],
                            'tpot_s': report['metrics_delta']['request_time_per_output_token_seconds']['mean'],
                            'acceptance_length': report['metrics_delta']['dspark']['mean_acceptance_length'],
                            'cache_hit_ratio': report['cache_hit_ratio'],
                        }), flush=True)
        h.save(root / 'complete.json', {'complete': True, 'time': time.time()})
    except BaseException:
        h.save(root / 'failure.json', {'error': traceback.format_exc(), 'time': time.time()})
        raise


if __name__ == '__main__':
    main()
