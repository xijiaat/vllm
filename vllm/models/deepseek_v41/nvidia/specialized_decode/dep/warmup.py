# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare DEP local kernels outside capture; never run EP collectives here."""

import torch

from vllm.config.compilation import (
    decode_cudagraph_capture_max_tokens,
    decode_cudagraph_capture_shapes,
)
from vllm.models.deepseek_v41.nvidia.ops.o_proj import (
    precompile_dsv41_o_proj_early_trigger,
)
from vllm.models.deepseek_v41.nvidia.specialized_decode.attention import (
    decode_topk_cute,
    mxfp8,
    sparse_mla_cute,
)
from vllm.models.deepseek_v41.nvidia.specialized_decode.boundary.boundary import (
    BoundaryState,
)
from vllm.models.deepseek_v41.nvidia.specialized_decode.logits.lm_head import (
    autotune_low_batch_lm_head,
    warmup_low_batch_lm_head,
)
from vllm.models.deepseek_v41.nvidia.specialized_decode.runtime.warmup import (
    _aux_weights,
    _decode_topk_variants,
    _tuning,
    _warm_up_mhc,
)
from vllm.v1.worker.gpu.spec_decode.dflash.config import DFlashSpeculatorConfig

from ..attention.aux_projections import (
    warmup_aux_projection_gemms,
)
from .shared_boundary import LocalMoEOutput


def prepare_rows(decode, rows):
    if not 0 < rows <= decode.config.max_rows:
        raise ValueError("DEP row capacity exceeded")
    with torch.inference_mode():
        for plan in decode.plans:
            attn = plan.attention.attn
            for linear in (attn.fused_wqa_wkv, attn.wq_b, attn.wo_b):
                if not mxfp8.mxfp8_eligible(linear):
                    continue
                if rows <= 8:
                    mxfp8.run_zeros(linear, rows)
                else:
                    decode.projections.prepare(linear, mxfp8._zeros(linear, rows))

        layer = decode.plans[0].layer
        device = layer.hc_attn_fn.device
        hidden, mult = layer.hidden_size, layer.hc_mult
        residual = torch.zeros(rows, mult, hidden, dtype=torch.bfloat16, device=device)
        pre = torch.zeros(rows, mult, dtype=torch.float32, device=device)
        pre[:, 0] = 1
        post = torch.zeros(rows, mult, 1, dtype=torch.float32, device=device)
        comb = (
            torch.eye(mult, dtype=torch.float32, device=device)
            .expand(rows, -1, -1)
            .contiguous()
        )
        pending = torch.zeros(rows, hidden, dtype=torch.bfloat16, device=device)
        state = BoundaryState(residual, post, comb, pre, pending)
        shared_state = BoundaryState(
            residual, post, comb, pre, LocalMoEOutput(pending, pending)
        )
        # Aux capture changes the fused kernel; prepare it before graph capture.
        outputs = []
        for quantize in (False, True):
            for capture_aux in (False, True):
                for value in (state, shared_state):
                    outputs.append(
                        decode.boundaries.sublayer(
                            layer,
                            False,
                            value,
                            quantize=quantize,
                            capture_aux=capture_aux,
                        )
                    )
        decode.boundaries.join()


def warm_up(decode, vllm_config, drafts=()):
    executors = [decode, *drafts]
    steps = [plan.attention for executor in executors for plan in executor.plans]
    linears = mxfp8.shape_representatives(
        linear
        for step in steps
        for linear in (step.attn.fused_wqa_wkv, step.attn.wq_b, step.attn.wo_b)
    )
    with torch.inference_mode():
        if decode.config.autotune:
            with _tuning(), mxfp8.pinned_tactics():
                for linear in linears:
                    mxfp8.run_zeros(linear, 8)
                autotune_low_batch_lm_head(decode.lm_head)
        for linear in linears:
            for rows in range(1, 9):
                mxfp8.run_zeros(linear, rows)
        warmup_aux_projection_gemms(_aux_weights(steps))
        warmup_low_batch_lm_head(decode.lm_head)
        device = torch.accelerator.current_device_index()
        for heads, sinks in {
            (s.attn.n_local_heads, s.attn.attn_sink is not None) for s in steps
        }:
            if heads in (16, 64):
                sparse_mla_cute.precompile(device, sinks, head_groups=heads // 16)
        for variant in _decode_topk_variants(steps):
            decode_topk_cute.precompile(device, *variant)
        for attn in {
            (s.attn.n_local_groups, s.attn.n_local_heads, s.attn.o_lora_rank): s.attn
            for s in steps
        }.values():
            precompile_dsv41_o_proj_early_trigger(attn, 48)
        for executor in executors:
            if decode.config.autotune:
                with _tuning():
                    executor.prepare_rows(executor.config.max_rows)
            else:
                executor.prepare_rows(executor.config.max_rows)
            # Use the runner's graph shapes; live eager fallback must not compile
            # unused specialized kernels for arbitrary prefill tail lengths.
            spec = vllm_config.speculative_config
            assert spec is not None
            query_len = spec.num_speculative_tokens + 1
            if executor.is_draft:
                speculator = spec.speculator_config
                assert isinstance(speculator, DFlashSpeculatorConfig)
                query_len = speculator.num_query_tokens
                query_lens = {query_len: vllm_config.scheduler_config.max_num_seqs}
            else:
                query_lens = vllm_config.get_decode_cudagraph_query_lens(query_len)
            compilation = vllm_config.compilation_config
            for num_reqs, query_len in decode_cudagraph_capture_shapes(
                compilation.decode_cudagraph_capture_request_sizes or [],
                query_lens,
                decode_cudagraph_capture_max_tokens(compilation),
            ):
                rows = num_reqs * query_len
                if 0 < rows <= executor.config.max_rows:
                    executor.prepare_rows(rows)
            if decode.config.jit_warmup:
                _warm_up_mhc(executor, vllm_config, executor.config.max_rows)
