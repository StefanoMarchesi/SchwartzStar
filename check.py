#!/usr/bin/env python3
"""Start one local server, record memory, TTFT, repetitions and compare two runs."""
import argparse
import hashlib
import http.client
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parent
HTTP = build_opener(ProxyHandler({}))
PROMPT = "The capitals of Italy and France are Rome and Paris. The capital of Spain is"
FILLER = (
    "An SSD keeps no moving parts, so a page read is a request to flash memory and not "
    "a seek on a spinning platter. The vsller maps logical pages to physical pages "
    "and erases whole blocks when it needs free space. "
)


def command_output(*args):
    return subprocess.run(args, capture_output=True, text=True, check=False).stdout.strip()


def swap_bytes():
    if sys.platform == "darwin":
        info = command_output("sysctl", "-n", "vm.swapusage")
        match = re.search(r"used = (\d+(?:\.\d+)?)([KMGT])", info)
        return round(float(match[1]) * 1024 ** ("KMGT".index(match[2]) + 1)) if match else None
    fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return (int(fields["SwapTotal"].split()[0]) - int(fields["SwapFree"].split()[0])) * 1024


def compressed_bytes():
    """Bytes of RAM occupied by the macOS compressor."""
    if sys.platform != "darwin":
        return None
    text = command_output("vm_stat")
    page = command_output("sysctl", "-n", "hw.pagesize")
    match = re.search(r"Pages occupied by compressor:\s+(\d+)", text)
    return int(match[1]) * int(page) if match and page.isdigit() else None


def memory_pressure():
    """System-wide free memory percentage, the figure macOS reports under pressure."""
    if sys.platform != "darwin":
        fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        return round(100.0 * int(fields["MemAvailable"].split()[0]) / int(fields["MemTotal"].split()[0]), 1)
    text = command_output("memory_pressure")
    match = re.search(r"System-wide memory free percentage:\s*(\d+)%", text)
    return int(match[1]) if match else None


def snapshot():
    compressed = compressed_bytes()
    swap = swap_bytes()
    return {
        "swap_mib": None if swap is None else round(swap / 1048576, 1),
        "compressed_mib": None if compressed is None else round(compressed / 1048576, 1),
        "memory_free_percent": memory_pressure(),
        "load_1m": round(os.getloadavg()[0], 2),
    }


def sample(pid, stop, memory, footprint):
    while not stop.is_set():
        rss = command_output("ps", "-o", "rss=", "-p", str(pid))
        if rss.isdigit():
            memory["peak_rss_mib"] = max(memory["peak_rss_mib"], int(rss) / 1024)
        if footprint:
            peak = command_output(str(footprint), str(pid))
            if peak.isdigit():
                memory["peak_physical_footprint_mib"] = max(
                    memory["peak_physical_footprint_mib"] or 0, int(peak) / 1048576
                )
        swap = swap_bytes()
        if swap is not None and memory["swap_initial_bytes"] is not None:
            memory["swap_peak_delta_mib"] = max(
                memory["swap_peak_delta_mib"], (swap - memory["swap_initial_bytes"]) / 1048576
            )
        stop.wait(0.25)


def query(base, route, body=None, timeout=5):
    data = None if body is None else json.dumps(body).encode()
    request = Request(base + route, data=data, headers={"Content-Type": "application/json"})
    with HTTP.open(request, timeout=timeout) as response:
        return json.load(response)


def stream_completion(base, body, timeout):
    """Post one streaming completion and time the first token that carries content."""
    host, port = base.replace("http://", "").split(":")
    connection = http.client.HTTPConnection(host, int(port), timeout=timeout)
    payload = json.dumps(dict(body, stream=True)).encode()
    started = time.monotonic()
    try:
        connection.request("POST", "/completion", body=payload, headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(f"Streaming /completion returned HTTP {response.status}")
        ttft = None
        content = []
        timings = None
        probabilities = None
        for raw in response:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            chunk = json.loads(line[5:].strip())
            piece = chunk.get("content") or ""
            if piece:
                content.append(piece)
                if ttft is None:
                    ttft = time.monotonic() - started
            if chunk.get("completion_probabilities"):
                probabilities = chunk["completion_probabilities"]
            if chunk.get("timings"):
                timings = chunk["timings"]
            if chunk.get("stop"):
                break
    finally:
        connection.close()
    return {
        "ttft_s": None if ttft is None else round(ttft, 3),
        "content": "".join(content),
        "timings": timings,
        "completion_probabilities": probabilities,
    }


def fixed_prefix(base, n_tokens):
    """Return a prompt of exactly n_tokens tokens, checked through the tokenizer."""
    ids = query(base, "/tokenize", {"content": FILLER}, timeout=60)["tokens"]
    if not ids:
        raise RuntimeError("The tokenizer returned no tokens")
    text = FILLER * (n_tokens // len(ids) + 2)
    ids = query(base, "/tokenize", {"content": text}, timeout=60)["tokens"][:n_tokens]
    if len(ids) != n_tokens:
        raise RuntimeError(f"Expected a {n_tokens}-token prefix, got {len(ids)}")
    return query(base, "/detokenize", {"tokens": ids}, timeout=60)["content"], n_tokens


def retrieval_prefix(base, n_tokens):
    """Place three independent records at known token offsets in a synthetic prefix."""
    expected = {"alder": "RV7KQ3", "birch": "J9MD4X", "cedar": "PT5N8B"}

    def tokenize(text):
        return query(base, "/tokenize", {"content": text}, timeout=60)["tokens"]

    filler = tokenize(FILLER)
    if not filler:
        raise RuntimeError("The tokenizer returned no filler tokens")
    prompt = tokenize("Use the verified inventory notes below to answer the final question.\n")
    offsets = {}
    for (name, value), numerator in zip(expected.items(), (1, 4, 7)):
        offset = n_tokens * numerator // 8
        fill = offset - len(prompt)
        prompt.extend((filler * (fill // len(filler) + 1))[:fill])
        offsets[name] = len(prompt)
        prompt.extend(tokenize(f"\nVerified inventory record: {name} has code {value}.\n"))
    question = tokenize("\nReturn the three inventory codes as JSON under the keys alder, birch, and cedar.\nAnswer:")
    fill = n_tokens - len(question) - len(prompt)
    if fill < 0:
        raise RuntimeError("The retrieval prefix is too short for the question and all records")
    prompt.extend((filler * (fill // len(filler) + 1))[:fill])
    return prompt + question, expected, offsets


def io_report(log_text):
    totals = re.search(
        r"Inkling SSD: reads = (\d+), read = (\d+(?:\.\d+)?) MiB, routed applications = (\d+)/(\d+)", log_text
    )
    phases = re.search(
        r"Inkling SSD I/O: prefill: tokens = (\d+), misses = (\d+), reads = (\d+), "
        r"read = (\d+(?:\.\d+)?) MiB", log_text
    )
    decode = re.search(
        r"Inkling SSD I/O: decode:\s+tokens = (\d+), misses = (\d+), reads = (\d+), "
        r"read = (\d+(?:\.\d+)?) MiB", log_text
    )
    times = re.search(
        r"Inkling SSD time: prefill: read = (\d+(?:\.\d+)?) ms, sync = (\d+(?:\.\d+)?) ms; "
        r"decode: read = (\d+(?:\.\d+)?) ms, sync = (\d+(?:\.\d+)?) ms", log_text
    )
    apps = re.search(r"Inkling SSD applications: lookups = (\d+), layers = (\d+), per layer = ([\d ]+)", log_text)
    unique = re.search(r"Inkling SSD: (\d+) unique expert tensors", log_text)
    layers = int(apps[2]) if apps else None
    # The cache counts once per layer application. Dividing by the number of layers gives the
    # evaluated tokens only if every application ran, which per_layer shows separately, so the
    # quotient is left unrounded: a fractional value means the coverage was incomplete.
    def evaluated(app_tokens):
        return None if app_tokens is None or not layers else app_tokens / layers

    return {
        "reads": int(totals[1]) if totals else None,
        "read_mib": float(totals[2]) if totals else None,
        "routed": f"{totals[3]}/{totals[4]}" if totals else None,
        "layers": layers,
        "prefill_app_tokens": int(phases[1]) if phases else None,
        "prefill_evaluated_tokens": evaluated(int(phases[1]) if phases else None),
        "prefill_misses": int(phases[2]) if phases else None,
        "prefill_reads": int(phases[3]) if phases else None,
        "prefill_mib": float(phases[4]) if phases else None,
        "decode_app_tokens": int(decode[1]) if decode else None,
        "decode_evaluated_tokens": evaluated(int(decode[1]) if decode else None),
        "decode_misses": int(decode[2]) if decode else None,
        "decode_reads": int(decode[3]) if decode else None,
        "decode_mib": float(decode[4]) if decode else None,
        "prefill_read_ms": float(times[1]) if times else None,
        "prefill_sync_ms": float(times[2]) if times else None,
        "decode_read_ms": float(times[3]) if times else None,
        "decode_sync_ms": float(times[4]) if times else None,
        "lookups": int(apps[1]) if apps else None,
        "per_layer": [int(v) for v in apps[3].split()] if apps else None,
        "unique_expert_tensors": int(unique[1]) if unique else None,
    }


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output / args.label
    model = args.model.expanduser().resolve()
    if not model.is_file():
        raise RuntimeError(f"Model file missing: {model}")
    env = dict(os.environ, INKLING_PORT=str(args.port))
    base = f"http://127.0.0.1:{args.port}"
    try:
        with HTTP.open(base + "/health", timeout=1):
            raise RuntimeError(f"Port already in use: {args.port}")
    except URLError as error:
        if not isinstance(error.reason, ConnectionRefusedError):
            raise RuntimeError(f"Port in use or unavailable for verification: {args.port}") from error

    footprint = None
    if sys.platform == "darwin":
        root = Path(env.get("INKLING_BUILD_DIR", ROOT / "build"))
        footprint = root / "bin" / "inkling-footprint"
        if not footprint.is_file():
            raise RuntimeError("Run ./build.sh to build the macOS footprint sampler")
    before = snapshot()
    memory = {
        "peak_rss_mib": 0, "peak_physical_footprint_mib": None,
        "swap_initial_bytes": swap_bytes(), "swap_peak_delta_mib": 0,
    }
    request_body = {
        "prompt": PROMPT, "n_predict": args.tokens, "temperature": 0,
        "seed": 1, "cache_prompt": False,
    }
    repetitions, prefix_probe, swa_probe, result = [], None, None, None
    stop = threading.Event()
    started = time.monotonic()
    with stem.with_suffix(".log").open("w") as log:
        proc = subprocess.Popen([str(ROOT / "serve.sh"), args.profile, str(model)], env=env, stdout=log, stderr=log)
        sampler = threading.Thread(target=sample, args=(proc.pid, stop, memory, footprint))
        sampler.start()
        try:
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f"Server exited. See {log.name}")
                try:
                    if query(base, "/health", timeout=1).get("status") == "ok":
                        break
                except (URLError, TimeoutError):
                    pass
                if time.monotonic() - started > args.timeout:
                    raise RuntimeError(f"Server startup timed out. See {log.name}")
                time.sleep(0.25)
            ready = time.monotonic() - started

            if args.prefix_only:
                # This run contains just one evaluation, so the phase counters apply to this prompt.
                count = args.prefix_only
                if args.retrieve:
                    prompt, expected, offsets = retrieval_prefix(base, count)
                    schema = {
                        "type": "object",
                        "properties": {key: {"type": "string"} for key in expected},
                        "required": list(expected), "additionalProperties": False,
                    }
                    body = dict(request_body, prompt=prompt, json_schema=schema)
                else:
                    prompt, count = fixed_prefix(base, count)
                    body = dict(request_body, prompt=prompt)
                probe = stream_completion(base, body, args.timeout)
                prefix_probe = {
                    "prompt_tokens": count,
                    "prompt_sha256": hashlib.sha256((json.dumps(prompt) if args.retrieve else prompt).encode()).hexdigest(),
                    "ttft_s": probe["ttft_s"],
                    "content": probe["content"], "timings": probe["timings"],
                }
                if args.retrieve:
                    try:
                        recovered = json.loads(probe["content"])
                    except json.JSONDecodeError:
                        recovered = None
                    prefix_probe["retrieval"] = {"record_offsets": offsets, "expected": expected,
                                                 "recovered": recovered, "passed": recovered == expected}
                repetitions.append({"request": "prefix-only", "repeat": 1, **probe})
                response = None
            else:
                # first request: streaming, timed from the request to the first token with content
                first = stream_completion(base, request_body, args.timeout)
                repetitions.append({"request": "stream", "repeat": 1, **first})
                # same generation again, non-streaming, for the per-token top-k probabilities
                result = query(base, "/completion", dict(request_body, n_probs=4,
                        post_sampling_probs=False), args.timeout)
                if result.get("content") != first["content"]:
                    raise RuntimeError("Streamed and non-streamed completions differ")
                for repeat in range(2, args.repeats + 1):
                    again = stream_completion(base, request_body, args.timeout)
                    repetitions.append({"request": "stream", "repeat": repeat, **again})
                    if again["content"] != first["content"]:
                        raise RuntimeError(f"Greedy completion changed on repetition {repeat}")
                response = query(base, "/v1/responses", {
                    "model": "inkling-2-mini", "input": "What is the capital of Italy?",
                    "max_output_tokens": 16, "temperature": 0, "stream": False,
                }, args.timeout)

            if args.prefix_tokens and not args.prefix_only:
                text, count = fixed_prefix(base, args.prefix_tokens)
                probe = stream_completion(base, dict(request_body, prompt=text, n_predict=8), args.timeout)
                prefix_probe = {
                    "prompt_tokens": count, "ttft_s": probe["ttft_s"],
                    "content": probe["content"], "timings": probe["timings"],
                }
            if args.swa_tokens:
                text, count = fixed_prefix(base, args.swa_tokens)
                probe = stream_completion(base, dict(request_body, prompt=text, n_predict=8), args.timeout)
                if not probe["content"]:
                    raise RuntimeError(f"No token generated past the SWA window ({count}-token prefix)")
                swa_probe = {
                    "prompt_tokens": count, "ttft_s": probe["ttft_s"],
                    "content": probe["content"], "timings": probe["timings"],
                }
        finally:
            if footprint and proc.poll() is None:
                peak = command_output(str(footprint), str(proc.pid))
                if peak.isdigit():
                    memory["peak_physical_footprint_mib"] = max(
                        memory["peak_physical_footprint_mib"] or 0, int(peak) / 1048576
                    )
            stop.set()
            sampler.join()
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
    after = snapshot()
    log_text = stem.with_suffix(".log").read_text(errors="replace")
    io = io_report(log_text)
    if args.profile != "resident":
        if not io["reads"]:
            raise RuntimeError("The log contains no SSD expert reads")
        if io["per_layer"] is None or len(io["per_layer"]) != io["layers"]:
            raise RuntimeError("The log is missing per-layer application counts")
        if io["routed"] and io["routed"].split("/")[0] != io["routed"].split("/")[1]:
            raise RuntimeError(f"Incomplete router application coverage: {io['routed']}")
        if not args.prefix_only and io["decode_reads"] == 0:
            raise RuntimeError("No expert reads during decode. Every lookup hit the cache")
    if not args.prefix_only:
        if not result.get("content") or not result.get("completion_probabilities"):
            raise RuntimeError("Completion text or probabilities missing")
        if not any(
            part.get("type") == "output_text" and part.get("text", "").strip()
            for item in response.get("output", []) if item.get("type") == "message"
            for part in item.get("content", [])
        ):
            raise RuntimeError("The Responses API returned no visible text")
    # tokens the server actually evaluated and emitted, kept apart from the cache counters
    counted = [step["timings"] for step in repetitions if step["timings"]]
    if result and result.get("timings"):
        counted.append(result["timings"])
    server_tokens = {
        "prompt_n": sum(int(t.get("prompt_n", 0)) for t in counted),
        "predicted_n": sum(int(t.get("predicted_n", 0)) for t in counted),
    }
    report = {
        "model": str(model), "profile": args.profile, "setup": {
            key: env.get(f"INKLING_{key}")
            for key in ("CTX", "BATCH", "CACHE_MIB", "CACHE_TYPE_K", "CTX_CHECKPOINTS",
                        "SSD_READ_CACHE", "THREADS", "SERVER")
        },
        "machine": {"before": before, "after": after},
        "startup_seconds": round(ready, 3), "memory": memory, "io": io, "server_tokens": server_tokens,
        "completion": result, "repetitions": repetitions,
        "prefix_probe": prefix_probe, "swa_probe": swa_probe, "responses_api": response,
    }
    path = stem.with_suffix(".json")
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "profile": report["profile"], "startup_seconds": report["startup_seconds"],
        "memory": memory, "machine": report["machine"], "io": io, "server_tokens": server_tokens,
    }, indent=2))
    for repeat in repetitions:
        timings = repeat["timings"] or {}
        print(
            f"request {repeat['repeat']}: TTFT {repeat['ttft_s']} s, "
            f"prefill {timings.get('prompt_ms')} ms for {timings.get('prompt_n')} token, "
            f"decode {timings.get('predicted_per_second')} tok/s"
        )
    print(path)
    if args.retrieve and (not prefix_probe["retrieval"]["passed"] or server_tokens["prompt_n"] != args.prefix_only):
        raise RuntimeError("The retrieval answer or evaluated token count did not match the expected result")
    if footprint:
        peak = memory["peak_physical_footprint_mib"]
        if peak is None:
            raise RuntimeError("Physical footprint unavailable for this run")
        if args.max_footprint_mib and peak > args.max_footprint_mib:
            raise RuntimeError(f"Physical footprint {peak:.1f} MiB exceeds the {args.max_footprint_mib} MiB target")


def probs_of(row):
    return {
        entry["id"]: entry.get("prob", math.exp(entry.get("logprob", -math.inf)))
        for entry in row["top_probs" if "top_probs" in row else "top_logprobs"]
    }


def compare(left, right, tolerance):
    """Report every divergence before failing, never stop at the first one."""
    a, b = [json.loads(path.read_text()) for path in (left, right)]
    if not Path(a["model"]).samefile(b["model"]):
        raise RuntimeError("Numerical parity requires the same GGUF")
    ca, cb = a["completion"], b["completion"]
    rows_a, rows_b = ca.get("completion_probabilities") or [], cb.get("completion_probabilities") or []
    if not rows_a or not rows_b:
        raise RuntimeError("The runs contain no top-k probabilities")
    differences = []
    if ca.get("content") != cb.get("content"):
        differences.append(f"different text: {ca.get('content')!r} vs {cb.get('content')!r}")
    worst, worst_at = 0.0, None
    for i, (x, y) in enumerate(zip(rows_a, rows_b)):
        if x.get("id") != y.get("id"):
            differences.append(
                f"position {i}: token {x.get('id')} {x.get('token')!r} vs {y.get('id')} {y.get('token')!r}"
            )
            continue
        if "top_probs" in x and "top_probs" in y:
            key = "top_probs"
        elif "top_logprobs" in x and "top_logprobs" in y:
            key = "top_logprobs"
        else:
            raise RuntimeError(f"Top-k probabilities missing at position {i}")
        pa, pb = probs_of(x), probs_of(y)
        if pa.keys() != pb.keys():
            differences.append(f"position {i}: top-k {sorted(pa)} vs {sorted(pb)}")
            continue
        for token, probability in pa.items():
            delta = abs(probability - pb[token])
            if delta > worst:
                worst, worst_at = delta, (i, token)
    if len(rows_a) != len(rows_b):
        differences.append(f"{len(rows_a)} token vs {len(rows_b)}")
    if differences:
        for line in differences[:5]:
            print(line)
        raise RuntimeError(f"{len(differences)} differences. Largest top-k delta: {worst:g} at position {worst_at}")
    if not math.isfinite(worst) or worst > tolerance:
        raise RuntimeError(f"Maximum delta {worst:g} at position {worst_at} exceeds tolerance {tolerance:g}")
    print(f"{len(rows_a)} matching greedy tokens. Largest absolute top-k delta: {worst:g} at position {worst_at}.")
    for name in ("prefill", "decode"):
        left_io, right_io = a["io"].get(f"{name}_mib"), b["io"].get(f"{name}_mib")
        if left_io is not None and right_io is not None:
            print(f"{name}: {left_io:.1f} MiB letti vs {right_io:.1f} MiB")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    trial = sub.add_parser("run")
    trial.add_argument("profile", choices=("coding", "quality", "minimum", "resident"))
    trial.add_argument("model", type=Path)
    trial.add_argument("label")
    trial.add_argument("--port", type=int, default=18080)
    trial.add_argument("--timeout", type=int, default=600)
    trial.add_argument("--tokens", type=int, default=24)
    trial.add_argument("--repeats", type=int, default=3, help="Consecutive greedy requests in the same server.")
    trial.add_argument("--prefix-tokens", type=int, default=0, help="Probe a fixed-token prefix.")
    trial.add_argument("--prefix-only", type=int, default=0, help="Benchmark a single request with this prefix length.")
    trial.add_argument("--retrieve", action="store_true", help="Recover three records in a fixed --prefix-only prompt.")
    trial.add_argument("--swa-tokens", type=int, default=0, help="Probe beyond the SWA window with this prefix length.")
    trial.add_argument("--max-footprint-mib", type=int, default=4500, help="macOS footprint threshold in MiB. Use 0 to disable.")
    trial.add_argument("--output", type=Path, default=Path("results"))
    parity = sub.add_parser("compare")
    parity.add_argument("left", type=Path)
    parity.add_argument("right", type=Path)
    parity.add_argument("--tolerance", type=float, default=0.005)
    args = parser.parse_args()
    if args.action == "run" and args.retrieve and args.prefix_only < 256:
        parser.error("--retrieve needs --prefix-only with at least 256 tokens")
    try:
        if args.action == "run":
            if not re.fullmatch(r"[\w-]+", args.label):
                parser.error("Use a label with letters, numbers, hyphens, or underscores")
            if args.repeats < 1 or args.tokens < 1:
                parser.error("Repetitions and generated tokens must be positive")
            run(args)
        else:
            compare(args.left, args.right, args.tolerance)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        sys.exit(str(error))


if __name__ == "__main__":
    main()
