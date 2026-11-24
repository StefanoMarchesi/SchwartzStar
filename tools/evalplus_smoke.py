#!/usr/bin/env python3
"""Generate and, in Docker, score a fixed three-task HumanEval+ smoke test.

This is deliberately not a full HumanEval+ score. Generation only calls a
loopback Chat Completions server and never executes completions. `--evaluate`
uses EvalPlus 0.3.1's `get_groundtruth` and `check_correctness` functions;
run that mode only in the pinned official Docker image printed after generation.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shlex
import time
from urllib.request import urlopen


TASK_IDS = ("HumanEval/0", "HumanEval/1", "HumanEval/2")
EVALPLUS_PACKAGE_VERSION = "0.3.1"
HUMANEVAL_PLUS_VERSION = "v0.1.10"
DATASET_URL = (
    "https://github.com/evalplus/humanevalplus_release/releases/download/"
    "v0.1.10/HumanEvalPlus.jsonl.gz"
)
IMAGE = "ganler/evalplus:v0.3.1"
SYSTEM = (
    "Complete the Python function in the user prompt. Return only the continuation "
    "after the supplied function signature/docstring, with no Markdown fence or explanation. "
    "Preserve Python indentation exactly."
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def stream_jsonl(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def strip_fence(text):
    """Remove only an enclosing Markdown fence; retain code indentation."""
    trimmed = text.strip()
    if trimmed.startswith("```"):
        lines = trimmed.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip("\r\n")
    return text.strip("\r\n")


def download_dataset(destination):
    print(f"Downloading pinned HumanEval+ dataset: {DATASET_URL}", flush=True)
    try:
        with urlopen(DATASET_URL, timeout=60) as source, destination.open("xb") as target:
            while chunk := source.read(1024 * 1024):
                target.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return destination


def load_tasks(dataset):
    tasks = {row["task_id"]: row for row in stream_jsonl(dataset)}
    required = {"task_id", "prompt", "contract", "canonical_solution", "base_input", "plus_input", "atol", "entry_point"}
    missing = [task_id for task_id in TASK_IDS if task_id not in tasks]
    invalid = [task_id for task_id in TASK_IDS if task_id in tasks and required - tasks[task_id].keys()]
    if missing or invalid:
        raise SystemExit(f"Dataset is not the expected HumanEval+ release; missing={missing}, invalid={invalid}")
    return {task_id: tasks[task_id] for task_id in TASK_IDS}


def evaluate(output):
    """Score only this subset through the pinned official EvalPlus APIs."""
    try:
        import evalplus
        from evalplus import evaluate as evaluator
        from evalplus.eval import PASS
        from evalplus.evaluate import check_correctness, get_groundtruth
    except ImportError as error:
        raise SystemExit(f"--evaluate requires EvalPlus {EVALPLUS_PACKAGE_VERSION}: {error}")
    installed = getattr(evalplus, "__version__", None)
    if installed and installed != EVALPLUS_PACKAGE_VERSION:
        raise SystemExit(f"Expected evalplus=={EVALPLUS_PACKAGE_VERSION}, found {installed}")
    subset = output / "humanevalplus-v0.1.10-smoke.jsonl.gz"
    samples_file = output / "samples.jsonl"
    if not subset.is_file() or not samples_file.is_file():
        raise SystemExit("--evaluate needs the generated subset and samples.jsonl")
    problems = load_tasks(subset)
    evaluator.CACHE_DIR = str(output / "oracle-cache")
    expected = get_groundtruth(problems, hashlib.md5(subset.read_bytes()).hexdigest(), [])
    samples = {sample["task_id"]: sample for sample in stream_jsonl(samples_file)}
    if set(samples) != set(TASK_IDS):
        raise SystemExit("samples.jsonl must contain exactly one completion for every disclosed smoke task")
    rows = []
    for task_id, problem in problems.items():
        solution = problem["prompt"] + samples[task_id]["completion"]
        result = check_correctness("humaneval", 0, problem, solution, expected[task_id], fast_check=False)
        base, _ = result["base"]
        plus, _ = result["plus"]
        rows.append({"task_id": task_id, "base_status": base, "plus_status": plus})
        print(f"{task_id}: base={base}, plus={plus}", flush=True)
    base_score = sum(row["base_status"] == PASS for row in rows) / len(rows)
    plus_score = sum(row["base_status"] == row["plus_status"] == PASS for row in rows) / len(rows)
    write_json(output / "evalplus-smoke-results.json", {
        "kind": "3-task HumanEval+ smoke subset; not a full-suite score",
        "base_pass_at_1": base_score, "plus_pass_at_1": plus_score, "tasks": rows,
    })
    print(f"Smoke pass@1: base={base_score:.3f}, HumanEval+={plus_score:.3f}")
    return 0


def generate(opts):
    if opts.output.exists():
        raise SystemExit(f"Output directory already exists: {opts.output}")
    opts.output.mkdir(mode=0o700, parents=True)
    if opts.dataset:
        dataset = opts.dataset.expanduser().resolve()
        if not dataset.is_file():
            raise SystemExit(f"--dataset is not a file: {dataset}")
        source = "user-supplied local file"
    else:
        dataset = download_dataset(opts.output / "HumanEvalPlus-v0.1.10.jsonl.gz")
        source = DATASET_URL
    problems = load_tasks(dataset)
    subset = opts.output / "humanevalplus-v0.1.10-smoke.jsonl.gz"
    with gzip.open(subset, "wt", encoding="utf-8") as handle:
        for problem in problems.values():
            handle.write(json.dumps(problem, ensure_ascii=False) + "\n")
    write_json(opts.output / "manifest.json", {
        "kind": "HumanEval+ smoke subset; not a full-suite score", "evalplus_package": EVALPLUS_PACKAGE_VERSION,
        "humaneval_plus_version": HUMANEVAL_PLUS_VERSION, "dataset_source": source,
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(), "task_ids": list(TASK_IDS),
        "decoding": {"temperature": 0, "top_p": 1, "n": 1, "max_tokens": opts.max_tokens}, "model": opts.model,
        "assistant_prefill": opts.prefill_text,
    })
    from php_agent_probe import request
    samples = []
    for index, task_id in enumerate(TASK_IDS, 1):
        body = {"model": opts.model, "messages": [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": problems[task_id]["prompt"]}], "stream": True,
                "stream_options": {"include_usage": True}, "temperature": 0, "top_p": 1, "n": 1,
                "max_tokens": opts.max_tokens, "cache_prompt": False, "reasoning_budget_tokens": 0,
                "chat_template_kwargs": {"enable_thinking": False}}
        if opts.prefill_text:
            body["messages"].append({"role": "assistant", "content": "    # Solution\n"})
            body["echo"] = True
        stem = opts.output / f"{index:02d}-{task_id.replace('/', '-')}"
        write_json(stem.with_suffix(".request.json"), body)
        started = time.monotonic()
        metric, completion, calls = request(opts.port, body, stem.with_suffix(".stream"))
        if calls:
            raise RuntimeError(f"{task_id}: unexpected tool calls; this benchmark accepts text only")
        metric.update(task_id=task_id, wall_s=time.monotonic() - started)
        write_json(stem.with_suffix(".metrics.json"), metric)
        samples.append({"task_id": task_id, "completion": strip_fence(completion)})
        print(f"Generated {task_id} ({metric['wall_s']:.1f}s)", flush=True)
    with (opts.output / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    runner, out = Path(__file__).resolve(), opts.output.resolve()
    print("\nGeneration complete. Evaluate stored completions only in the pinned isolated container:")
    print("docker run --rm --network none " +
          f"-v {shlex.quote(str(out))}:/work:rw -v {shlex.quote(str(runner))}:/runner/evalplus_smoke.py:ro " +
          f"-w /work {IMAGE} python /runner/evalplus_smoke.py --evaluate /work")
    print("This is a disclosed three-task smoke pass@1, not a full HumanEval+ score.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New output directory (or an existing one with --evaluate)")
    parser.add_argument("--evaluate", action="store_true", help="Score stored samples with official EvalPlus APIs")
    parser.add_argument("--model", help="Local Chat Completions model name (required unless --evaluate)")
    parser.add_argument("--port", type=int, default=18080, help="Loopback Chat Completions port")
    parser.add_argument("--max-tokens", type=int, default=512, help="Greedy cap per task")
    parser.add_argument("--dataset", type=Path, help="Official release asset instead of downloading it")
    parser.add_argument("--prefill-text", action="store_true", help="Open Inkling's text block with an indented Python comment")
    opts = parser.parse_args()
    if opts.evaluate:
        return evaluate(opts.output.resolve())
    if not opts.model:
        parser.error("--model is required unless --evaluate")
    if opts.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    return generate(opts)


if __name__ == "__main__":
    raise SystemExit(main())
