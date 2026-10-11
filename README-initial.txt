H20 C1/C2 compressor/cache fusion: initial operator evidence

All 26 shapes and all 50 AB/BA pairs per shape are in result.json. This is one
process/run, not three independent model campaigns. CUDA-event timing includes
128 host-launched graph replays per sample; device-batched graph replay and
model-level validation are pending. No end-to-end speedup is claimed.

The baseline is vLLM 1ee69dde9cdfc949cfae896016b44357ca864a50; its compressor file
was also byte-identical to main when checked on 2026-10-11. candidate.py is the
exact GPU-tested source. One H20-3e, CUDA SM90, torch 2.13.0+cu130. No model
weights are loaded by this harness.

Reproduction in the same vLLM/CUDA environment on an H20:
  mount this directory read-only at /source in a one-GPU container
  uv venv --system-site-packages --python /usr/bin/python3 /results/.venv
  /results/.venv/bin/python /source/validate_bench.py --out /results/result.json

Validation first executes the two relevant definitions extracted from the
existing test_compressor_kv_cache.py reference test (40 combinations), without
running unrelated module imports/conftest. It then compares baseline/candidate
state, latent and full paged cache bytes across 18 cases and five graph replays,
including changed/restored inputs and poisoned reused output. The full pytest
module was not run. No existing numerical tolerance was relaxed.

This evidence branch is separate from the small code PR. The code PR remains a
Draft without serving integration, NCU evidence or model accuracy/performance
results. SGLang #41660 by DarkSharpness is the fusion-boundary reference; the
prototype preserves vLLM's original FP32 pooling/RMSNorm and BF16 rounding.
