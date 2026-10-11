"""GPU-batched graph timing and separate NCU ranges; no model or service."""
import argparse
import json
from pathlib import Path
import statistics
import time
import torch
import validate_bench as h


def capture_chain(case, module, arm, repeats):
    def body():
        for _ in range(repeats):
            case.call(module, arm)
    return h.graph(body)


def sample(graph, repeats):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / repeats


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--ncu', action='store_true')
    args=parser.parse_args()
    torch.manual_seed(42)
    assert torch.cuda.device_count()==1 and torch.cuda.get_device_capability()==(9,0)
    base=h.load('batch_baseline','/source/baseline.py')
    cand=h.load('batch_candidate','/source/candidate.py')
    import vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache as installed
    assert h.sha(installed.__file__)==h.sha('/source/baseline.py')
    out={'status':'RUNNING','graph_nodes_per_replay':128,'torch':torch.__version__,
         'script_sha256':h.sha(__file__),'baseline_sha256':h.sha('/source/baseline.py'),
         'candidate_sha256':h.sha('/source/candidate.py'),'started':time.time(),
         'method':'50 alternating AB/BA pairs; each CUDA-event sample exactly one128-chain graph replay; no host gap between nodes',
         'benchmark':[]}
    path=Path('/results/batched.json')
    if args.ncu:
        cases=[]
        for ratio in (1,2):
            for mode,lengths in [('decode',[1]),('verify6',[6]*16),('prefill',[16384])]:
                case=h.Case(ratio,lengths,pad=0)
                for _ in range(100):
                    case.call(base,0);case.call(cand,1)
                case.compare()
                cases.append((ratio,mode,case))
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        for ratio,mode,case in cases:
            for arm,module in enumerate((base,cand)):
                torch.cuda.nvtx.range_push(f'C{ratio}_{mode}_'+('split' if arm==0 else 'fused'))
                case.call(module,arm)
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()
        Path('/results/ncu-input-proof.json').write_text(json.dumps(out,indent=2))
        return
    for ratio in (1,2):
        shapes=[(mode,[length]*r) for r in (1,4,16,64,128) for mode,length in [('decode',1),('verify6',6)]]
        shapes += [('prefill',[m]) for m in (1024,4096,16384)]
        for mode,lengths in shapes:
            case=h.Case(ratio,lengths,pad=0)
            gs=[capture_chain(case,module,arm,128) for arm,module in enumerate((base,cand))]
            case.reset()
            for g in gs:g.replay()
            case.compare()
            for _ in range(20):
                for g in gs:g.replay()
            torch.cuda.synchronize()
            pairs=[]
            for pair in range(50):
                order=(0,1) if pair%2==0 else (1,0)
                times={str(arm):sample(gs[arm],128) for arm in order}
                pairs.append({'pair':pair,'order':order,'us':times})
            a=statistics.median([row['us']['0'] for row in pairs])
            b=statistics.median([row['us']['1'] for row in pairs])
            record={'ratio':ratio,'mode':mode,'requests':len(lengths),'M':sum(lengths),
                    'baseline_median_us':a,'candidate_median_us':b,'speedup':a/b,'pairs':pairs}
            out['benchmark'].append(record)
            path.write_text(json.dumps(out,indent=2))
            print('DEVICE_BATCHED',ratio,mode,len(lengths),sum(lengths),a,b,a/b,flush=True)
    out['status']='PASS';out['finished']=time.time()
    path.write_text(json.dumps(out,indent=2))

if __name__=='__main__':main()
