"""Fixed real-model A/B workloads; imports only unmodified HTTP/metric helpers."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import random
import re
import statistics
import time
from decimal import Decimal
import bench_helpers as h

client = h.Client('http://127.0.0.1:18751', timeout=900)
ROOT = Path('/results/bench')
ROOT.mkdir(exist_ok=False)
PLAN = json.loads(Path('/code/plan.json').read_text())
h.save(ROOT/'plan.json', PLAN)


def settle():
    for _ in range(100):
        raw = client.call('/metrics').decode()
        if h.is_idle(raw):
            time.sleep(0.3)
            return client.call('/metrics').decode()
        time.sleep(0.2)
    raise AssertionError('not idle')


def reset():
    settle()
    assert json.loads(client.call('/reset_prefix_cache', {}))['success'] is True


def prompt_rows():
    path = Path('/shared/prompts.json')
    if path.exists():
        rows = json.loads(path.read_text())
    else:
        rows = []
        for j in range(32):
            rng = random.Random(713+j)
            text = f'Ledger {j}. Read the records.\n' + ''.join(
                f'Record {i}: warehouse {rng.randrange(10000)}, received {rng.randrange(1000)}, shipped {rng.randrange(500)}; stock inspected and checked.\n'
                for i in range(1800))
            text += '\nSummarize the inventory records in detail.'
            ids = json.loads(client.call('/tokenize', {'model':'dsv41','prompt':text,'add_special_tokens':False}))['tokens']
            assert len(ids) > 16384
            rows.append(ids[:16384-64]+ids[-64:])
        h.save(path, rows)
    assert len(rows)==32 and all(len(row)==16384 for row in rows)
    h.save(ROOT/'input-proof.json', {'sha256': h.sha(path), 'sessions':32, 'source':'fixed-ledger-seed713'})
    return rows


def prime(rows, out):
    out.mkdir()
    hits = []
    for p in range(2):
        for i, row in enumerate(rows):
            response = json.loads(client.call('/v1/completions', {'model':'dsv41','prompt':row,
                'add_special_tokens':False,'max_tokens':1,'temperature':0,'top_p':1,'seed':713,'ignore_eos':True}))
            h.save(out/f'pass{p}-{i}.json', response)
            assert response['usage']['prompt_tokens']==len(row)
            if p==1:
                hits.append(response['usage']['prompt_tokens_details']['cached_tokens'])
    assert len(set(hits))==1 and 0<hits[0]<len(rows[0]), hits
    settle()
    h.save(out/'observed-hit.json', {'hits':hits,'note':'observed, not inferred maximum; report residual prefill'})
    return hits[0]


def hot(rows):
    for length, c in PLAN['hot_cells']:
        reset()
        selected = [row[:length-64]+row[-64:] for row in rows[:2*c]]
        cell = ROOT/f'hot-{length}-c{c}'
        cell.mkdir()
        hit = prime(selected, cell/'priming')
        for i in range(PLAN['hot_calibrations']+PLAN['hot_samples']):
            phase = 'calibration' if i<PLAN['hot_calibrations'] else 'sample'
            out = cell/f'{phase}-{i:02d}'
            out.mkdir()
            report = h.measure(client, selected,length,c,'dsv41',True,out,hit)
            # The shared helper historically labelled the value as source-derived.
            # This campaign records it as independently observed priming only.
            report['full_warm_means_engine_supported_limit_not_100_percent'] = False
            report['cache_hit_source'] = 'second priming pass, observed'
            for request in report['requests']:
                request.pop('expected_max_cached_tokens', None)
            (out/'sample.json').write_text(json.dumps(report,indent=2))
            print('HOT',length,c,phase,report['output_tok_s'],report['metrics_delta']['dspark']['mean_acceptance_length'],flush=True)


def cold(rows):
    for length,c in PLAN['cold_cells']:
        cell=ROOT/f'cold-{length}-c{c}'
        cell.mkdir()
        selected=[row[:length-64]+row[-64:] for row in rows[:2*c]]
        for i in range(PLAN['cold_calibrations']+PLAN['cold_samples']):
            reset()
            phase='calibration' if i<PLAN['cold_calibrations'] else 'sample'
            out=cell/f'{phase}-{i:02d}'
            out.mkdir()
            def one(pair):
                j,row=pair
                begin=time.perf_counter()
                response=json.loads(client.call('/v1/completions', {'model':'dsv41','prompt':row,
                    'add_special_tokens':False,'max_tokens':1,'temperature':0,'top_p':1,'seed':713,'ignore_eos':True}))
                elapsed=time.perf_counter()-begin
                usage=response['usage']
                assert usage['prompt_tokens']==length and usage['completion_tokens']==1
                assert usage['prompt_tokens_details']['cached_tokens']==0, usage
                return {'index':j,'duration_s':elapsed,'response':response}
            before=settle()
            start=time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=c) as pool:
                responses=list(pool.map(one,enumerate(selected)))
            duration=time.perf_counter()-start
            after=settle()
            counters=h.metric_deltas(before,after,False)
            assert counters['request_success_total']==2*c
            assert counters['prompt_tokens_total']==2*c*length
            assert counters['generation_tokens_total']==2*c
            assert counters['num_preemptions_total']==0
            report={'requests':responses,'duration_s':duration,'input_tok_s':2*c*length/duration,
                'metrics_delta':counters,'metrics_before':before,'metrics_after':after,
                'cache_hit_tokens':0,'length':length,'concurrency':c,'output_tokens_per_request':1}
            h.save(out/'sample.json',report)
            print('COLD',length,c,phase,report['input_tok_s'],flush=True)


def number(text):
    matches=re.findall(r'####\s*([-+]?[\d,]+(?:\.\d+)?)',text.rsplit('</think>',1)[-1])
    return str(Decimal(matches[-1].replace(',','')).normalize()) if matches else None


def quality():
    reset()
    ds=Path('/code/gsm8k')
    manifest=json.loads((ds/'manifest.json').read_text())
    for name,record in manifest['files'].items():
        assert h.sha(ds/name)==record['sha256']
    train=[json.loads(x) for x in (ds/'train.jsonl').read_text().splitlines()]
    test=[json.loads(x) for x in (ds/'test.jsonl').read_text().splitlines()][:128]
    prefix=''.join(f"Question: {x['question']}\nAnswer: {x['answer']}\n\n" for x in train[:5])
    out=ROOT/'quality'; out.mkdir()
    h.save(out/'plan.json', {'dataset':manifest,'indices':list(range(128)),**PLAN['quality'],
                            'apc':True,'note':'paired regression subset, not full GSM8K'})
    def one(pair):
        i,row=pair
        prompt=prefix+f"Question: {row['question']}\nAnswer:"
        response=json.loads(client.call('/v1/chat/completions',{'model':'dsv41','messages':[{'role':'user','content':prompt}],
            'temperature':0,'top_p':1,'seed':713,'max_tokens':1024,'chat_template_kwargs':{'thinking':False}}))
        text=response['choices'][0]['message'].get('content') or ''
        result={'index':i,'response':response,'answer':number(text),'expected':number(row['answer']),
                'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()}
        result['correct']=result['answer'] is not None and result['answer']==result['expected']
        h.save(out/f'{i:03d}.json',result)
        return result
    before=settle()
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        responses=list(pool.map(one,enumerate(test)))
    after=settle()
    h.save(out/'summary.json', {'questions':128,'correct':sum(x['correct'] for x in responses),
        'length_finished':sum(x['response']['choices'][0]['finish_reason']=='length' for x in responses),
        'metrics':h.metric_deltas(before,after,True), 'subset_only':True})
    print('QUALITY',sum(x['correct'] for x in responses),128,flush=True)

rows=prompt_rows()
quality()
hot(rows)
cold(rows)
h.save(ROOT/'complete.json',{'completed':True,'time':time.time(),'plan_sha256':h.sha('/code/plan.json')})
