"""One arm's fixed independently prewarmed-prefix decode grid; never launches/stops an engine.

Adapted from exp24 bench_model.py: same ledger/token-ID construction and
continuous workers, now streamed 1024-token output with real metrics deltas.
"""

import argparse
import base64
import concurrent.futures
import hashlib
import http.client
import json
import math
from pathlib import Path
import random
import re
import statistics
import threading
import time
import traceback
from urllib.parse import urlsplit

CELLS = [(length, c) for length in (4096, 16384, 65536) for c in (1, 4, 8, 16, 32)]
OUTPUT_TOKENS = 1024


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    with Path(path).open("x") as handle:
        json.dump(obj, handle, indent=2)


def values(text, name):
    pattern = re.compile(r"^" + re.escape(name) + r"(?:\{[^\n]*\})?\s+([^\s]+)(?:\s|$)")
    return [
        float(match.group(1))
        for line in text.splitlines()
        if (match := pattern.match(line))
    ]


def total(text, name, required=True):
    items = values(text, name)
    if required:
        assert items, f"missing metric {name}"
    return sum(items)


def delta(before, after, name, required=True):
    result = total(after, name, required) - total(before, name, required)
    assert result >= 0, ("counter reset", name, result)
    return result


def idle(text):
    assert total(text, "vllm:num_requests_running") == 0, (
        "server already running requests"
    )
    assert total(text, "vllm:num_requests_waiting") == 0, (
        "server already waiting requests"
    )


def is_idle(text):
    return (
        total(text, "vllm:num_requests_running") == 0
        and total(text, "vllm:num_requests_waiting") == 0
    )


def metric_deltas(before, after, dspark):
    keys = [
        "prompt_tokens_total",
        "generation_tokens_total",
        "request_success_total",
        "num_preemptions_total",
    ]
    result = {key: delta(before, after, "vllm:" + key) for key in keys}
    for prefix in [
        "time_to_first_token_seconds",
        "request_time_per_output_token_seconds",
        "e2e_request_latency_seconds",
    ]:
        count = delta(before, after, "vllm:" + prefix + "_count", required=False)
        summed = delta(before, after, "vllm:" + prefix + "_sum", required=False)
        result[prefix] = {
            "count": count,
            "sum": summed,
            "mean": summed / count if count else None,
        }
    spec = {
        key: delta(before, after, "vllm:spec_decode_" + key + "_total", required=dspark)
        for key in ["num_drafts", "num_draft_tokens", "num_accepted_tokens"]
    }
    spec["accepted_fraction"] = (
        spec["num_accepted_tokens"] / spec["num_draft_tokens"]
        if spec["num_draft_tokens"]
        else None
    )
    spec["mean_acceptance_length"] = (
        1 + spec["num_accepted_tokens"] / spec["num_drafts"]
        if spec["num_drafts"]
        else None
    )
    # Labels and per-position counters stay in both unmodified raw snapshots.
    spec["per_position_raw_before"] = [
        line
        for line in before.splitlines()
        if line.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total{")
    ]
    spec["per_position_raw_after"] = [
        line
        for line in after.splitlines()
        if line.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total{")
    ]

    def per_position(lines):
        positions = {}
        for line in lines:
            position = re.search(r'(?:\{|,)position="(\d+)"', line)
            assert position is not None, line
            index = position.group(1)
            positions[index] = positions.get(index, 0) + float(
                line.rsplit("}", 1)[1].strip().split()[0]
            )
        return positions

    old, new = (
        per_position(spec["per_position_raw_before"]),
        per_position(spec["per_position_raw_after"]),
    )
    assert set(old) == set(new), (
        "DSpark position metrics appeared/disappeared within sample"
    )
    spec["per_position"] = {
        position: {
            "accepted_tokens": new[position] - old[position],
            "fraction_of_drafts": (new[position] - old[position]) / spec["num_drafts"]
            if spec["num_drafts"]
            else None,
        }
        for position in new
    }
    assert all(row["accepted_tokens"] >= 0 for row in spec["per_position"].values())
    if dspark:
        assert old, "DSpark declared but position metrics absent"
        assert spec["num_accepted_tokens"] <= spec["num_draft_tokens"]
    else:
        assert (
            spec["num_drafts"]
            == spec["num_draft_tokens"]
            == spec["num_accepted_tokens"]
            == 0
        ), "non-speculative arm observed speculative work"
    result["dspark"] = spec
    return result


class Client:
    def __init__(self, base, timeout=600):
        parsed = urlsplit(base)
        assert parsed.scheme == "http" and parsed.hostname in ("127.0.0.1", "localhost")
        assert not parsed.path and not parsed.query and parsed.username is None
        self.host, self.port, self.timeout = parsed.hostname, parsed.port or 80, timeout

    def connection(self):
        return http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)

    def call(self, path, payload=None):
        connection = self.connection()
        try:
            body = None if payload is None else json.dumps(payload).encode()
            connection.request(
                "GET" if body is None else "POST",
                path,
                body,
                {"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            raw = response.read()
            assert response.status == 200, (response.status, raw[:300])
            return raw
        finally:
            connection.close()


def prepare_prompts(client, path, model):
    if not path.exists():
        rows = []
        for j in range(18):
            rng = random.Random(713 + j)
            text = f"Ledger {j}. Read the records.\n"
            text += "".join(
                f"Record {i}: warehouse {rng.randrange(10000)}, received {rng.randrange(1000)}, shipped {rng.randrange(500)}; stock inspected and checked.\n"
                for i in range(1800)
            )
            text += "\nSummarize the inventory records in one sentence."
            ids = json.loads(
                client.call(
                    "/tokenize",
                    {"model": model, "prompt": text, "add_special_tokens": False},
                )
            )["tokens"]
            assert len(ids) >= 16384
            rows.append(ids[: 16384 - 64] + ids[-64:])
        save(path, rows)
    rows = json.loads(path.read_text())
    assert len(rows) == 18 and all(len(row) >= 16384 for row in rows)
    assert all(
        all(isinstance(token, int) and token >= 0 for token in row) for row in rows
    )
    return rows


def stream_request(
    connection,
    payload,
    index,
    expected_output=OUTPUT_TOKENS,
    *,
    untimed_priming=False,
    priming_raw_path=None,
):
    assert not untimed_priming or expected_output == 1
    assert (priming_raw_path is not None) == untimed_priming
    start = time.perf_counter()
    connection.request(
        "POST", "/v1/completions", payload, {"Content-Type": "application/json"}
    )
    response = connection.getresponse()
    assert response.status == 200, (response.status, response.read()[:300])
    first = last = None
    content, chunks, usage, finish, done = [], [], None, None, False
    raw_lines = [] if untimed_priming else None
    try:
        while True:
            line = response.readline()
            if raw_lines is not None:
                raw_lines.append(line)
            if not line:
                break
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                done = True
                break
            event = json.loads(data)
            assert "error" not in event, event
            if event.get("usage") is not None:
                usage = event["usage"]
            for choice in event.get("choices", []):
                text = choice.get("text") or ""
                if text:
                    now = time.perf_counter()
                    if first is None:
                        first = now
                    last = now
                    content.append(text)
                    chunks.append({"seconds": now - start, "characters": len(text)})
                finish = choice.get("finish_reason") or finish
        # Drain HTTP chunk terminator before persistent-connection reuse.
        remainder = response.read()
        if raw_lines is not None:
            raw_lines.append(remainder)
    finally:
        if priming_raw_path is not None:
            save(
                priming_raw_path,
                {
                    "untimed_priming": True,
                    "expected_output_tokens": expected_output,
                    "sse_body_base64": base64.b64encode(b"".join(raw_lines)).decode(),
                    "done_observed": done,
                    "usage_observed": usage,
                    "finish_reason_observed": finish,
                    "visible_text_observed": first is not None,
                },
            )
    end = time.perf_counter()
    assert done and usage is not None
    assert untimed_priming or (first is not None and last is not None)
    assert usage["completion_tokens"] == expected_output and finish == "length", (
        usage,
        finish,
    )
    cached = usage.get("prompt_tokens_details", {}).get("cached_tokens")
    assert isinstance(cached, int) and 0 <= cached < usage["prompt_tokens"], usage
    return {
        "index": index,
        "duration_s": end - start,
        "client_ttft_s": first - start if first is not None else None,
        "client_mean_tpot_s": (last - first) / (usage["completion_tokens"] - 1)
        if usage["completion_tokens"] > 1
        else None,
        "stream_content_chunks": chunks,
        "last_content_s": last - start if last is not None else None,
        "output_sha256": hashlib.sha256("".join(content).encode()).hexdigest(),
        "usage": usage,
        "finish_reason": finish,
        "done": done,
    }


def measure(
    client,
    rows,
    length,
    concurrency,
    model,
    dspark,
    output,
    expected_cached,
    session_offset=0,
    session_metadata=None,
):
    selected = [
        rows[(index + session_offset) % len(rows)] for index in range(2 * concurrency)
    ]
    assert all(len(row) == length for row in selected)
    payloads = [
        json.dumps(
            {
                "model": model,
                "prompt": row,
                "max_tokens": OUTPUT_TOKENS,
                "temperature": 0,
                "top_p": 1,
                "seed": 713,
                "ignore_eos": True,
                "add_special_tokens": False,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
        ).encode()
        for row in selected
    ]
    before = client.call("/metrics").decode()
    idle(before)
    save(output / "before.json", {"unix": time.time(), "metrics": before})
    index = iter(enumerate(payloads))
    lock = threading.Lock()
    barrier = threading.Barrier(concurrency + 1, timeout=60)
    completed = []

    def worker():
        connection = client.connection()
        connection.connect()
        results = []
        barrier.wait()
        try:
            while True:
                with lock:
                    try:
                        i, payload = next(index)
                    except StopIteration:
                        break
                result = stream_request(connection, payload, i)
                with lock:
                    completed.append(result)
                assert result["usage"]["prompt_tokens"] == length
                result["session_index"] = (i + session_offset) % len(rows)
                if session_metadata is not None:
                    result["session"] = session_metadata[result["session_index"]][
                        "session"
                    ]
                    result["scenario"] = session_metadata[result["session_index"]][
                        "scenario"
                    ]
                result["expected_observed_cached_tokens"] = expected_cached
                result["residual_prompt_tokens"] = (
                    length - result["usage"]["prompt_tokens_details"]["cached_tokens"]
                )
                assert (
                    result["usage"]["prompt_tokens_details"]["cached_tokens"]
                    == expected_cached
                ), ("prefix not at independently observed priming value", result)
                results.append(result)
        finally:
            connection.close()
        return results

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(worker) for _ in range(concurrency)]
            start = time.perf_counter()
            barrier.wait()
            requests = [item for future in futures for item in future.result()]
            duration = time.perf_counter() - start
    finally:
        # Never write per-request files between timed requests. Preserve completed
        # requests on failure too, after all outstanding workers have finished.
        for result in sorted(completed, key=lambda row: row["index"]):
            save(output / f"request-{result['index']:02d}.json", result)
    assert len(requests) == 2 * concurrency
    for attempt in range(30):
        after = client.call("/metrics").decode()
        counters = metric_deltas(before, after, dspark)
        if (
            is_idle(after)
            and counters["request_success_total"] >= len(requests)
            and counters["generation_tokens_total"] >= OUTPUT_TOKENS * len(requests)
        ):
            break
        time.sleep(0.2)
    save(output / "after.json", {"unix": time.time(), "metrics": after})
    idle(after)
    assert counters["request_success_total"] == len(requests), (
        "foreign traffic or incomplete counters",
        counters,
    )
    assert counters["prompt_tokens_total"] == length * len(requests), counters
    assert counters["generation_tokens_total"] == OUTPUT_TOKENS * len(requests), (
        counters
    )
    if dspark:
        assert counters["dspark"]["num_drafts"] > 0, (
            "DSpark declared but no real draft activity"
        )
    assert counters["num_preemptions_total"] == 0, (
        "preemption invalidates decode-only cache regime",
        counters,
    )
    report = {
        "length": length,
        "concurrency": concurrency,
        "output_tokens_per_request": OUTPUT_TOKENS,
        "requests": sorted(requests, key=lambda row: row["index"]),
        "duration_s": duration,
        "input_tok_s": length * len(requests) / duration,
        "output_tok_s": OUTPUT_TOKENS * len(requests) / duration,
        "metrics_delta": counters,
        "cache_hit_tokens": expected_cached * len(requests),
        "cache_hit_ratio": expected_cached / length,
        "residual_prompt_tokens_per_request": length - expected_cached,
        "observed_priming_hit_reproduced": True,
        "client_request_statistics": {},
        "note": "mixed full-request throughput, not pure prefill peak; client TTFT is first nonempty text chunk; multiple tokens per SSE chunk can bias client TPOT, so server metric deltas are separately retained",
    }
    for field in ("client_ttft_s", "client_mean_tpot_s", "duration_s"):
        data = sorted(row[field] for row in requests)
        report["client_request_statistics"][field] = {
            "all": data,
            "median": statistics.median(data),
            "mean": statistics.mean(data),
            "p95_nearest_rank": data[max(0, math.ceil(0.95 * len(data)) - 1)],
            "min": data[0],
            "max": data[-1],
        }
    save(output / "sample.json", report)
    return report


def cache_limit(geometry, length):
    """Restricted source-derived limit, never fitted to measured cache hits.

    a982 coordinator compensates draft-group drop while finding the aligned
    intersection. For all cacheable FullAttentionSpec groups with equal supported
    physical alignment, the last-token cap is round_down(P-1, A). Bounded-replay
    SWA is explicitly non-cacheable and its replay work is additional, not billed
    as an extra reduction of usage.cached_tokens. Unknown geometry fails closed.
    """
    assert len(geometry["groups"]) > 1, (
        "Unitary cache has a different Eagle drop contract"
    )
    alignment = geometry["cache_hit_alignment_tokens"]
    assert isinstance(alignment, int) and alignment > 0 and length % alignment == 0
    full = []

    def inspect(spec):
        if spec.get("inner_specs"):
            for inner in spec["inner_specs"].values():
                inspect(inner)
            return
        if not spec["prefix_cacheable"]:
            return
        assert spec["is_full_attention"], ("unsupported cacheable geometry", spec)
        assert alignment % spec["block_size"] == 0, (alignment, spec)
        full.append(spec)

    for group in geometry["groups"]:
        inspect(group["spec"])
    assert full, "No cacheable full-attention groups"
    result = (length - 1) // alignment * alignment
    assert result > 0, "This context cannot have a nonzero aligned cache hit"
    return result


def prime(client, rows, length, model, expected, output):
    """Two complete untimed passes, including per-request exact-hit verification."""
    before = client.call("/metrics").decode()
    idle(before)
    save(output / "before.json", {"unix": time.time(), "metrics": before})
    connection = client.connection()
    results = []
    try:
        for pass_index in range(2):
            for index, row in enumerate(rows):
                payload = json.dumps(
                    {
                        "model": model,
                        "prompt": row,
                        "max_tokens": 1,
                        "temperature": 0,
                        "top_p": 1,
                        "seed": 713,
                        "ignore_eos": True,
                        "add_special_tokens": False,
                        "stream": True,
                        "stream_options": {"include_usage": True},
                    }
                ).encode()
                result = stream_request(
                    connection,
                    payload,
                    index,
                    expected_output=1,
                    untimed_priming=True,
                    priming_raw_path=output / f"pass{pass_index}-session{index:02d}.sse.json",
                )
                result["pass"] = pass_index
                result["expected_observed_cached_tokens"] = expected
                save(output / f"pass{pass_index}-session{index:02d}.json", result)
                assert result["usage"]["prompt_tokens"] == length
                results.append(result)
                if pass_index == 1:
                    assert (
                        result["usage"]["prompt_tokens_details"]["cached_tokens"]
                        == expected
                    ), (
                        "prewarm below source-derived maximum; refusing to label fully warm",
                        result,
                    )
    finally:
        connection.close()
    for attempt in range(50):
        after = client.call("/metrics").decode()
        success = delta(before, after, "vllm:request_success_total")
        if is_idle(after) and success >= len(results):
            break
        time.sleep(0.2)
    save(output / "after.json", {"unix": time.time(), "metrics": after})
    idle(after)
    assert success == len(results)
    assert delta(before, after, "vllm:generation_tokens_total") == len(results)
    assert delta(before, after, "vllm:num_preemptions_total") == 0
    save(
        output / "complete.json",
        {
            "requests": len(results),
            "verified_cached_tokens": expected,
            "residual_prompt_tokens": length - expected,
        },
    )


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads(args.metadata.read_text())
    assert metadata["enable_prefix_caching"] is True
    assert (
        metadata["speculative_method"] == "dspark"
        and metadata["num_speculative_tokens"] == 5
    )
    assert metadata["tensor_parallel_size"] == 4 and len(metadata["gpu_uuids"]) == 4
    assert metadata["arm"] in ("A", "B")
    assert sha(args.prompts) == metadata["prompt_ledger_sha256"]
    ledger = json.loads(args.prompts.read_text())
    assert ledger["lengths"] == [4096, 16384, 65536] and len(ledger["prompts"]) == 18
    geometry = json.loads(args.cache_geometry.read_text())
    expected = {length: cache_limit(geometry, length) for length in ledger["lengths"]}
    client = Client(args.base)
    save(
        args.output / "plan.json",
        {
            "cells": CELLS,
            "output_tokens": OUTPUT_TOKENS,
            "samples": args.samples,
            "calibrations": 2,
            "requests_per_trial": "2*concurrency; continuous refill",
            "session_mapping": "18 sessions interleaved inventory/code/Chinese; trial offset=(trial ordinal within calibration/measurement)*2*c modulo18; c16/c32 intentionally reuse warmed sessions",
            "session_metadata": [
                {"session": row["session"], "scenario": row["scenario"]}
                for row in ledger["prompts"]
            ],
            "prompt_construction": ledger["construction"],
            "metadata": metadata,
            "metadata_sha256": sha(args.metadata),
            "prompts_sha256": sha(args.prompts),
            "script_sha256": sha(__file__),
            "cache_geometry": geometry,
            "expected_observed_cached_tokens": expected,
            "cache_policy": "Each length prewarmed independently per arm: all 18 prompts twice, second pass must reach source-derived maximum; every timed/calibration request must match exactly",
            "full_warm_is_not_100_percent": True,
        },
    )
    result = []
    try:
        for length in ledger["lengths"]:
            rows = [row["token_ids"][str(length)] for row in ledger["prompts"]]
            assert all(
                len(row) == length
                and all(type(token) is int and token >= 0 for token in row)
                for row in rows
            )
            warm = args.output / f"prime-{length}"
            warm.mkdir()
            prime(client, rows, length, args.model, expected[length], warm)
            for concurrency in (1, 4, 8, 16, 32):
                cell = []
                for ordinal in range(2 + args.samples):
                    phase = "calibration" if ordinal < 2 else "measurement"
                    output = args.output / f"{length}-c{concurrency}-{phase}-{ordinal}"
                    output.mkdir()
                    report = measure(
                        client,
                        rows,
                        length,
                        concurrency,
                        args.model,
                        True,
                        output,
                        expected[length],
                        session_offset=(
                            ordinal if phase == "calibration" else ordinal - 2
                        )
                        * 2
                        * concurrency,
                        session_metadata=ledger["prompts"],
                    )
                    if phase == "measurement":
                        cell.append(report["output_tok_s"])
                    print(
                        json.dumps(
                            {
                                "event": "SAMPLE",
                                "length": length,
                                "concurrency": concurrency,
                                "phase": phase,
                                "ordinal": ordinal,
                                "output_tok_s": report["output_tok_s"],
                                "cached_tokens_per_request": expected[length],
                                "residual_prompt_tokens": length - expected[length],
                            }
                        ),
                        flush=True,
                    )
                result.append(
                    {
                        "length": length,
                        "concurrency": concurrency,
                        "all_output_tok_s": cell,
                        "median": statistics.median(cell),
                        "stdev": statistics.stdev(cell),
                    }
                )
        save(
            args.output / "complete.json",
            {
                "status": "COMPLETE_ONE_ARM",
                "cells": result,
                "no_AB_gain_claim": True,
                "full_warm_is_not_100_percent": True,
            },
        )
    except BaseException:
        save(
            args.output / "failure.json",
            {"error": traceback.format_exc(), "completed_cells": result},
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:18751")
    parser.add_argument("--model", default="dsv41")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--cache-geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()
    assert args.samples == 3, "Frozen hot-prefix plan has exactly three measured trials"
    main(args)
