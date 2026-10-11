"""Compare the immutable two-launch vLLM baseline with optional 584B fusion."""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
import time

import torch
import vllm
from vllm.platforms import current_platform


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def graph(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    result = torch.cuda.CUDAGraph()
    with torch.cuda.graph(result):
        fn()
    return result


class Case:
    def __init__(self, ratio, lengths, pad=5, sparse_slots=False, raw_pad=0):
        self.ratio, self.lengths = ratio, lengths
        self.actual = sum(lengths)
        self.num = self.actual + pad
        self.capacity = 8
        nreq = len(lengths)
        starts = [7 + (i % 5) for i in range(nreq)]
        self.positions = torch.zeros(self.num, dtype=torch.int64, device="cuda")
        self.slots = torch.full((self.num,), -1, dtype=torch.int64, device="cuda")
        self.cache_slots = torch.full_like(self.slots, -1)
        self.req_ids = torch.zeros(self.num, dtype=torch.int32, device="cuda")
        self.query = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()],
                                  dtype=torch.int32, device="cuda")
        cursor = 0
        for r, (start, length) in enumerate(zip(starts, lengths)):
            idx = slice(cursor, cursor + length)
            positions = torch.arange(start, start + length, device="cuda")
            self.positions[idx] = positions
            self.slots[idx] = (nreq - r) * 8 + positions % 8
            self.req_ids[idx] = r
            self.cache_slots[idx] = torch.arange(cursor, cursor + length,
                                                 device="cuda").flip(0)
            cursor += length
        if sparse_slots and self.actual:
            self.cache_slots[:self.actual:3] = -1
        self.raw_backing = torch.randn(self.num, 512 * ratio + raw_pad, device="cuda")
        self.raw = self.raw_backing[:, :512 * ratio]
        self.norm = torch.randn(512, dtype=torch.bfloat16, device="cuda")
        angles = torch.randn(max(starts) + max(lengths, default=0) + 32, 32,
                             device="cuda")
        self.cs = torch.cat([angles.cos(), angles.sin()], dim=-1)
        self.state_init = torch.randn(nreq + 2, 8208, device="cuda")
        self.block = 256 // ratio
        self.stride = math.ceil(self.block * 584 / 576) * 576
        pages = max(1, math.ceil(self.actual / self.block)) + 1
        self.sides = []
        for _ in range(2):
            sb = self.state_init.clone()
            cb = torch.full((pages, self.stride), 165, dtype=torch.uint8, device="cuda")
            latent = torch.full((self.num, 512), float("nan"),
                                dtype=torch.bfloat16, device="cuda")
            self.sides.append(dict(sb=sb, cb=cb, latent=latent,
                state=sb.as_strided((nreq+2, 8, 1024), (8208, 1024, 1)),
                cache=cb.as_strided((pages, self.block, 584), (self.stride, 584, 1))))

    def call(self, mod, arm):
        s = self.sides[arm]
        kwargs = {} if arm == 0 else dict(
            cos_sin_cache=self.cs, kv_cache=s["cache"],
            cache_slot_mapping=self.cache_slots)
        mod.fused_save_compress_norm(
            self.raw, self.positions, s["state"] if self.ratio == 2 else None,
            self.slots, self.query if self.ratio == 2 else None,
            self.req_ids if self.ratio == 2 else None,
            self.norm, 1e-20, self.ratio, s["latent"], **kwargs)
        if arm == 0:
            mod.rope_quant_insert(s["latent"], self.positions, self.cs,
                                  s["cache"], self.cache_slots, self.ratio)

    def reset(self):
        for side in self.sides:
            side["sb"].copy_(self.state_init)
            side["cb"].fill_(165)
            side["latent"].fill_(float("nan"))

    def compare(self):
        for key in ("sb", "cb", "latent"):
            a, b = [s[key].view(torch.uint8) for s in self.sides]
            if not torch.equal(a, b):
                count = torch.count_nonzero(a != b).item()
                raise AssertionError(f"CR{self.ratio} {self.lengths}: {key}: {count} bytes differ")


def correctness(base, cand, out):
    # Reuse the real upstream reference test, extracted without its unrelated
    # package/conftest imports. Its numerical assertions are unchanged.
    src = Path('/source/test_compressor_kv_cache.py').read_text()
    tree = ast.parse(src)
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in ('test_v41_fused_save_compress_and_insert', '_ue8m0_reference')]
    assert len(selected) == 2
    for node in selected:
        node.decorator_list = []
    import pytest
    namespace = dict(torch=torch, math=math, pytest=pytest, current_platform=current_platform)
    key = 'vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache'
    original = sys.modules.get(key)
    sys.modules[key] = cand
    try:
        exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])),
                     'extracted_upstream_regression', 'exec'), namespace)
        reference_count = 0
        for ratio in (1, 2):
            for use_graph in (False, True):
                for fused in (False, True):
                    for lengths in ((1,5,13), (1,1,1), (1,2,0), (7,8,8), (2,3,3)):
                        namespace['test_v41_fused_save_compress_and_insert'](
                            ratio, use_graph, fused, lengths)
                        reference_count += 1
    finally:
        if original is None:
            del sys.modules[key]
        else:
            sys.modules[key] = original
    print('INDEPENDENT_REFERENCE_PASS', reference_count, flush=True)
    records = []
    for ratio in (1, 2):
        for lengths in ((0,0,0), (1,1,1), (1,2,0), (7,8,8), (2,3,3),
                        (6,6,6), (6,)*128, (16384,), (1023,1024,1)):
            c = Case(ratio, lengths, sparse_slots=True, raw_pad=16)
            gs = [graph(lambda a=a, m=m: c.call(m, a)) for a, m in enumerate((base,cand))]
            raw0, state0, pos0 = c.raw.clone(), c.state_init.clone(), c.positions.clone()
            slots0 = c.slots.clone()
            for replay in range(4):
                if replay == 3:
                    c.raw.copy_(raw0)
                    c.state_init.copy_(state0)
                    c.positions.copy_(pos0)
                    c.slots.copy_(slots0)
                elif replay:
                    c.raw.normal_()
                    c.state_init.normal_()
                    c.positions[:c.actual].add_(2)
                    c.slots[:c.actual] = slots0[:c.actual] // 8 * 8 + c.positions[:c.actual] % 8
                c.reset()
                for g in gs:
                    g.replay()
                c.compare()
            # Shrink the valid batch inside the same graph and poison reused output.
            keep = c.actual // 2
            c.slots[keep:] = -1
            c.cache_slots[keep:] = -1
            c.reset()
            for g in gs:
                g.replay()
            c.compare()
            records.append(dict(ratio=ratio,lengths=lengths,replays=5,
                                bytewise_state_latent_cache='PASS'))
            print('BYTEWISE_PASS', ratio, len(lengths), sum(lengths), flush=True)
    out['correctness'] = dict(reference_cases=reference_count, strict_cases=records)


def elapsed_us(g, repeats=128):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        g.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / repeats


def benchmark(base, cand, out, path):
    matrix = []
    for ratio in (1, 2):
        for r in (1,4,16,64,128):
            matrix.append((ratio, 'decode', [1]*r))
            matrix.append((ratio, 'verify6', [6]*r))
        for length in (1024,4096,16384):
            matrix.append((ratio, 'prefill', [length]))
    out['benchmark'] = []
    for ratio, mode, lengths in matrix:
        c = Case(ratio, lengths, pad=0)
        gs = [graph(lambda a=a, m=m: c.call(m,a)) for a,m in enumerate((base,cand))]
        c.reset()
        for g in gs:
            g.replay()
        c.compare()
        for _ in range(500):
            for g in gs:
                g.replay()
        torch.cuda.synchronize()
        samples = []
        for pair in range(50):
            order = (0,1) if pair % 2 == 0 else (1,0)
            pair_times = {}
            for arm in order:
                pair_times[str(arm)] = elapsed_us(gs[arm])
            samples.append(dict(pair=pair,order=order,us=pair_times))
        a = [s['us']['0'] for s in samples]
        b = [s['us']['1'] for s in samples]
        ma, mb = statistics.median(a), statistics.median(b)
        record = dict(ratio=ratio,mode=mode,requests=len(lengths),M=sum(lengths),
                      baseline_median_us=ma,candidate_median_us=mb,
                      speedup=ma/mb,latency_change_pct=(mb/ma-1)*100,
                      baseline_range=[min(a),max(a)],candidate_range=[min(b),max(b)],
                      pairs=samples)
        out['benchmark'].append(record)
        path.write_text(json.dumps(out,indent=2))
        print('BENCH', ratio, mode, len(lengths), sum(lengths),
              round(ma,4), round(mb,4), round(ma/mb,4), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='/results/result.json')
    p.add_argument('--mode', choices=['all','correctness','ncu'], default='all')
    args = p.parse_args()
    path = Path(args.out)
    torch.manual_seed(42)
    base = load('compress_base', '/source/baseline.py')
    cand = load('compress_candidate', '/source/candidate.py')
    import vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache as installed
    out = dict(start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        torch=torch.__version__, vllm=vllm.__version__, gpu=torch.cuda.get_device_name(),
        capability=torch.cuda.get_device_capability(), source={
            'baseline':sha('/source/baseline.py'), 'candidate':sha('/source/candidate.py'),
            'harness':sha(__file__), 'installed':sha(installed.__file__)},
        methodology='50 AB/BA pairs per shape; 500 calibration replays; each sample 128 graph replays; CUDA events')
    assert out['capability'] == (9,0)
    assert out['source']['baseline'] == out['source']['installed'], out['source']
    try:
        if args.mode == 'ncu':
            c = Case(2, [6]*128, pad=0)
            c.call(base,0)
            c.call(cand,1)
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            c.call(base,0)
            c.call(cand,1)
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            return
        correctness(base,cand,out)
        path.write_text(json.dumps(out,indent=2))
        if args.mode == 'all':
            benchmark(base,cand,out,path)
        out['status'] = 'PASS'
    except BaseException as e:
        out['status'], out['exception'] = 'FAIL', repr(e)
        raise
    finally:
        out['end_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
        out['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        path.write_text(json.dumps(out,indent=2))
    print('FINISHED', out['status'], flush=True)


if __name__ == '__main__':
    main()
