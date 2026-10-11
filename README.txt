DeepSeek-V4.1 C1/C2 compressor/cache fusion evidence for vLLM PR61088

Code PR head: 4fe427bd137145c3cd9a8a12c0517d53b533976f
Baseline: 1ee69dde9cdfc949cfae896016b44357ca864a50
The three unmodified runtime files were also byte-identical to upstream main
93d2cab35e8ae66bbaa1dea1c077986d5a79eb80 when checked on2026-10-11.

Root result.json, validate_bench.py, baseline.py, candidate.py and the original
test module are the initial operator evidence, unchanged. See README-initial.txt
for that run's exact protocol and limitations. The kernel source did not change
when adding serving integration.

integration/ contains the independently device-batched graph harness and summary,
plus the updated committed test. All46selected full-module pytest cases passed
on one H20. The graph harness captures128complete chains in one graph, avoiding
host submission gaps between those chains;20calibration graphs,50AB/BApairs per
shape,26shapes. All measurements and correctness checks remain separate from
model throughput. Reused buffers are L2-warm. NCU evidence has not been collected;
the --ncu harness branch is prepared but unexecuted.

With this evidence directory mounted at /source in a matching BASELINE image:
  uv venv --system-site-packages --python /usr/bin/python3 /results/.venv
  PYTHONPATH=/source /results/.venv/bin/python /source/integration/batched_ncu.py

With the PR installed, run its committed existing regression functions:
  .venv/bin/python -m pytest tests/kernels/test_compressor_kv_cache.py \
    --confcutdir=tests/kernels \
    -k 'test_v41_fused_save_compress_and_insert or test_v41_attention_joins_cache_writes_before_consumption'

model/REPORT.txt contains the complete initial real-model A/B interpretation,
including all timed throughput samples, acceptance, memory and limitations.
The fixed plan and exact HTTP benchmark/metadata helpers are included. The
benchmark expects them at /code, a writable /results and a shared /shared input
directory across arms, plus a loopback-only server at127.0.0.1:18751. Populate
/code/gsm8k from the pinned public URLs and hashes in its manifest. That manifest
describes the full source dataset; the actual test selection is the first128
test rows, fixed in plan.json. No private model weights or operational launcher
are included. Copy commands/server settings are in REPORT.txt.

The original ####-only score is preserved. The documented supplementary numeric
boxed-answer extractor was frozen before candidate B measurements. This is a
paired regression subset, not a full model quality evaluation. One complete
model A/B pair does not establish a stable causal throughput gain.

Full new device-batched/model raw archives remain on the test host; their hashes
are recorded in the summaries. They are not included in this evidence branch.
All raw samples from the initial operator run ARE included in root result.json.

The fusion boundary is inspired by SGLang#41660 byDarkSharpness. The implementation
retains vLLM's own FP32 pooling/RMSNorm, BF16 latent rounding, state ring and
584-byte FP8/BF16 cache representation. No model weights, Humming backend or
cache allocation changed. AI assistance: Codex. Keep the code PR as Draft.
