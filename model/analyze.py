"""Report every fixed A/B cell; retain correctness/acceptance differences."""
import argparse
import hashlib
import json
from decimal import Decimal
from pathlib import Path
import re
import statistics


def read(path):
    return json.loads(path.read_text())


def describe(samples):
    return {'all':samples,'median':statistics.median(samples),'min':min(samples),'max':max(samples),
            'relative_stdev_pct':statistics.stdev(samples)/statistics.mean(samples)*100 if len(samples)>1 else None}


def explicit_answer(text):
    """Supplement the original #### score with explicit numeric boxed answers.

    Frozen before candidate measurements. No last-number heuristic or reference-
    dependent extraction. The original score is retained and reported separately.
    """
    text = text.rsplit('</think>', 1)[-1]
    marked = re.findall(r'####\s*([-+]?[\d,]+(?:\.\d+)?)', text)
    boxed = re.findall(r'\\boxed\{\s*([-+]?[\d,]+(?:\.\d+)?)\s*\}', text)
    matches = marked or boxed
    return str(Decimal(matches[-1].replace(',', '')).normalize()) if matches else None


def main(root):
    result={'status':'INITIAL_AB_ONLY','model_speedup_stable':False,'cells':[], 'quality':{}}
    arms={arm:root/name/'raw' for arm,name in [('A','0A'),('B','1B')]}
    for arm,path in arms.items():
        assert read(path/'bench/complete.json')['completed']
        result[arm]={'plan':read(path/'bench/plan.json'), 'import_proof':read(path/'import-proof.json'),
                    'input_proof':read(path/'bench/input-proof.json'),
                    'inventory':[read(p) for p in sorted((path/'inventory').glob('rank-*.json'))]}
    assert result['A']['plan']==result['B']['plan']
    assert result['A']['input_proof']==result['B']['input_proof']
    assert result['A']['import_proof']['tilelang_seed_files']==result['B']['import_proof']['tilelang_seed_files']
    assert len(result['A']['inventory'])==len(result['B']['inventory'])==4
    for a,b in zip(result['A']['inventory'],result['B']['inventory']):
        assert a['rank']==b['rank'] and a['gpu_uuid']==b['gpu_uuid']
        for role in ('target','draft'):
            assert a['models'][role]['linear_kernels']==b['models'][role]['linear_kernels']
            assert a['models'][role]['moe_methods']==b['models'][role]['moe_methods']
    cells=sorted(p.name for p in (arms['A']/'bench').iterdir() if p.name.startswith(('hot-','cold-')))
    assert len(cells)==5
    for cell in cells:
        before=cell.startswith('hot-')
        field='output_tok_s' if before else 'input_tok_s'
        reports={}
        for arm,path in arms.items():
            paths=sorted((path/'bench'/cell).glob('sample-*/sample.json'))
            assert len(paths)==3,(arm,cell,len(paths))
            rows=[read(p) for p in paths]
            reports[arm]={'throughput':describe([r[field] for r in rows]),'all_records':rows}
            if before:
                reports[arm]['acceptance_length']=describe([r['metrics_delta']['dspark']['mean_acceptance_length'] for r in rows])
                reports[arm]['tpot_s']=describe([r['metrics_delta']['request_time_per_output_token_seconds']['mean'] for r in rows])
                reports[arm]['ttft_s']=describe([r['metrics_delta']['time_to_first_token_seconds']['mean'] for r in rows])
        a,b=(reports[arm]['throughput']['median'] for arm in ('A','B'))
        entry={'cell':cell,'field':field,'percent_change':(b/a-1)*100,**reports}
        result['cells'].append(entry)
        print(cell,round(a,3),round(b,3),round(entry['percent_change'],3))
    quality={}
    for arm,path in arms.items():
        quality[arm]={r['index']:r for r in (read(p) for p in sorted((path/'bench/quality').glob('[0-9][0-9][0-9].json')))}
        assert len(quality[arm])==128
        result['quality'][arm]=read(path/'bench/quality/summary.json')
    comparisons=[]
    for i in range(128):
        a,b=quality['A'][i],quality['B'][i]
        assert a['expected']==b['expected'] and a['prompt_sha256']==b['prompt_sha256']
        comparisons.append({'index':i,'answer_match':a['answer']==b['answer'],
            'content_match':a['response']['choices'][0]['message']==b['response']['choices'][0]['message'],
            'baseline_correct':a['correct'],'candidate_correct':b['correct']})
    result['quality']['comparison']=comparisons
    result['quality']['answer_matches']=sum(r['answer_match'] for r in comparisons)
    result['quality']['content_matches']=sum(r['content_match'] for r in comparisons)
    result['quality']['subset_only']=True
    supplemental = {'rule':'last #### number, else last explicit numeric \\boxed{number}; no heuristic fallback',
                    'original_scores_preserved':True, 'rows':[]}
    for i in range(128):
        row = {'index':i}
        for arm in ('A','B'):
            source = quality[arm][i]
            answer = explicit_answer(source['response']['choices'][0]['message'].get('content') or '')
            row[arm] = {'answer':answer, 'parsed':answer is not None,
                        'correct':answer is not None and answer == source['expected']}
        row['answer_match'] = row['A']['parsed'] and row['B']['parsed'] and row['A']['answer']==row['B']['answer']
        supplemental['rows'].append(row)
    for arm in ('A','B'):
        supplemental[arm] = {key:sum(r[arm][key] for r in supplemental['rows']) for key in ('parsed','correct')}
    supplemental['answer_matches'] = sum(r['answer_match'] for r in supplemental['rows'])
    result['quality']['explicit_format_supplement'] = supplemental
    output=root/'analysis.json';output.write_text(json.dumps(result,indent=2)+'\n')
    print('QUALITY',result['quality']['A']['correct'],result['quality']['B']['correct'],result['quality']['answer_matches'],result['quality']['content_matches'])
    print('EXPLICIT_FORMAT_SUPPLEMENT',supplemental['A'],supplemental['B'],supplemental['answer_matches'])
    print('REPORT',output,hashlib.sha256(output.read_bytes()).hexdigest())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
