#!/usr/bin/env python3
"""Benchmark a local coding loop with structured patches and sandboxed tests."""

import argparse
import difflib
import json
from pathlib import Path
import shutil
import subprocess
import time

from php_agent_probe import FILES, request, run_tests, snapshot, test_passed
from php_fixture import SHORT_CONTEXT, TURN_1, TURN_2, populate


EDIT_SCHEMA = {
    "type": "object",
    "properties": {field: {"type": "string"} for field in ("old", "new")},
    "required": ["old", "new"],
    "additionalProperties": False,
}
PATCH_SCHEMA = {
    "type": "object",
    "properties": {name: EDIT_SCHEMA for name in ("invoice", "costs")},
    "required": ["invoice", "costs"],
    "additionalProperties": False,
}
APPEND_SCHEMA = {
    "type": "object",
    "properties": {name: {"type": "string"} for name in ("invoice", "costs")},
    "required": ["invoice", "costs"],
    "additionalProperties": False,
}
SYSTEM = (
    "You edit a synthetic PHP repository. Return only compact JSON, without markdown. "
    "Keep the existing public APIs. The controller applies edits and runs tests."
)
EDIT_INSTRUCTIONS = (
    "Return keys invoice and costs, each containing old and new strings. "
    "Copy the smallest unique old snippets verbatim, including whitespace."
)
APPEND_INSTRUCTIONS = (
    "Return keys invoice and costs, each containing only the new function definition as a string. "
    "The controller appends each string to its existing file. Omit PHP tags. "
    "The costs function can call the new invoice function directly."
)


def read_sources(root):
    return {name: (root / name).read_text(encoding="utf-8") for name in FILES}


def update_sources(sources, answer, append):
    patch = json.loads(answer)
    result = {}
    for name, text in sources.items():
        key = Path(name).stem
        edit = patch[key]
        if append:
            if not edit.strip() or "<?" in edit or "?>" in edit:
                raise ValueError(f"{key}: return only the function definition without PHP tags")
            result[name] = f"{text.rstrip()}\n\n{edit.strip()}\n"
            continue
        old, new = edit["old"], edit["new"]
        if not old or text.count(old) != 1 or old == new:
            raise ValueError(f"{key}: provide a unique exact old snippet and a different new snippet")
        result[name] = text.replace(old, new, 1)
    return result


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_turn(opts, php, root, messages, turn, task, baseline):
    sources = read_sources(root)
    files = "\n\n".join(f"{name}\n```php\n{text}```" for name, text in sources.items())
    append = turn == 2
    context = "" if append else SHORT_CONTEXT + "\n\n"
    instructions = APPEND_INSTRUCTIONS if append else EDIT_INSTRUCTIONS
    schema = APPEND_SCHEMA if append else PATCH_SCHEMA
    messages.append({"role": "user", "content": f"{context}{task}\n\n{instructions}\n\n{files}"})
    before = snapshot()
    started = time.monotonic()
    attempts = []
    passed = False
    for step in range(1, opts.attempts + 1):
        body = {
            "model": opts.model, "messages": messages, "stream": True,
            "stream_options": {"include_usage": True}, "temperature": 0,
            "max_tokens": opts.max_tokens, "cache_prompt": turn > 1 or step > 1,
            "reasoning_effort": "none", "reasoning_budget_tokens": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "php_edits", "strict": True, "schema": schema,
            }},
        }
        if opts.prefill_json:
            body["messages"] = [*messages, {"role": "assistant", "content": "{"}]
            body["echo"] = True
        stem = opts.output / f"turn-{turn}-request-{step:02}"
        save_json(stem.with_suffix(".json"), body)
        print(f"Structured patch, turn {turn}, attempt {step}", flush=True)
        metric, answer, _ = request(opts.port, body, stem.with_suffix(".stream"))
        try:
            replacements = update_sources(sources, answer, append)
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            metric["patch_error"] = str(error)
            feedback = f"Patch rejected. {error}"
        else:
            for name, replacement in replacements.items():
                (root / name).write_text(replacement, encoding="utf-8")
            metric["test"] = run_tests(php, root, turn)
            metric["regression"] = run_tests(php, root, 1) if turn == 2 else metric["test"]
            passed = test_passed(metric["test"], turn) and test_passed(metric["regression"], 1)
            feedback = f"Tests: {metric['test']}. Turn 1 regression: {metric['regression']}."
        attempts.append(metric)
        save_json(stem.with_suffix(".metrics.json"), metric)
        messages.append({"role": "assistant", "content": answer})
        if passed:
            break
        for name, text in sources.items():
            (root / name).write_text(text, encoding="utf-8")
        messages.append({"role": "user", "content": (
            f"{feedback}\nThe controller restored both files to the versions supplied at the "
            "start of this turn. Return a corrected JSON patch for those versions."
        )})

    elapsed = time.monotonic() - started
    final = read_sources(root)
    diff = "".join(line for name in FILES for line in difflib.unified_diff(
        baseline[name].splitlines(keepends=True), final[name].splitlines(keepends=True),
        fromfile=f"a/{name}", tofile=f"b/{name}",
    ))
    (opts.output / f"turn-{turn}.diff").write_text(diff, encoding="utf-8")
    footprint = int(subprocess.check_output([str(opts.footprint), str(opts.pid)], text=True)) / 1048576
    report = {
        "turn": turn, "passed": passed, "time_to_green_s": elapsed if passed else None,
        "total_s": elapsed, "footprint_mib": round(footprint, 1), "requests": attempts,
        "before": before, "after": snapshot(),
    }
    save_json(opts.output / f"turn-{turn}.json", report)
    print(f"Turn {turn}: {'PASS' if passed else 'FAIL'}, {elapsed:.1f} s", flush=True)
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New directory for the fixture and raw trace")
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--footprint", type=Path, required=True)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--prefill-json", action="store_true", help="Start the assistant's content block with '{'")
    opts = parser.parse_args()
    if opts.attempts < 1 or opts.max_tokens < 1:
        parser.error("--attempts and --max-tokens must be positive")
    php = shutil.which("php")
    if not php:
        parser.error("php is required")
    opts.output.mkdir(mode=0o700)
    root = opts.output.resolve() / "fixture"
    root.mkdir()
    populate(root)
    baseline_test = run_tests(php, root, 1)
    save_json(opts.output / "baseline.json", baseline_test)
    if baseline_test["exit"] != 1 or baseline_test["output"].strip() != "FAIL: invoice breakdown contract":
        raise RuntimeError(f"Unexpected initial test result: {baseline_test}")
    baseline = read_sources(root)
    messages = [{"role": "system", "content": SYSTEM}]
    for turn, task in enumerate((TURN_1, TURN_2), 1):
        if not run_turn(opts, php, root, messages, turn, task, baseline):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
