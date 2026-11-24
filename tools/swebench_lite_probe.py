#!/usr/bin/env python3
"""Run a bounded guided edit on one frozen Flask case and score it offline."""

import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import textwrap
import time

from php_agent_probe import request
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check import query

SOURCE = Path("src/flask/config.py")
SOURCE_SHA256 = "51ba3f8eecd262c00aa83daf0f7be6ea68ec3e8dc43890dd484cd8abc94a4c92"
TEST_PATCH_SHA256 = "cc1c3da8b7230ccecdeab7cecb230347343a603b519f59af442c0bf3d1db6f87"
PROBLEM = """# Coding task

Modify `repo/src/flask/config.py` to make `Config.from_file` usable with both
text-mode loaders (the existing behavior, such as `json.load`) and binary-mode
loaders (such as Python 3.11+'s `tomllib.load`).

Add a backward-compatible optional parameter that lets callers choose text or
binary opening. It must default to the existing text behavior, while the
binary option must pass a binary file object to the loader. Preserve the
existing return value, path handling, missing-file `silent` behavior, and
error handling. Keep the change focused on `Config.from_file`, and update its
docstring/API documentation as appropriate.

Do not change unrelated behavior. The relevant base implementation is included
in `context.txt`. Implement the change in the checkout under `repo/`."""
SCHEMA = {"type": "object", "properties": {"replacement": {"type": "string"}},
          "required": ["replacement"], "additionalProperties": False}
EDIT = {
    "type": "object",
    "properties": {"start": {"type": "integer"}, "end": {"type": "integer"},
                   "replacement": {"type": "string", "maxLength": 500}},
    "required": ["start", "end", "replacement"],
    "additionalProperties": False,
}
EDIT_SCHEMA = {
    "type": "object",
    "properties": {"edits": {"type": "array", "items": EDIT, "minItems": 1, "maxItems": 4}},
    "required": ["edits"],
    "additionalProperties": False,
}


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def method_span(source):
    config = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == "Config")
    method = next(node for node in config.body if isinstance(node, ast.FunctionDef) and node.name == "from_file")
    return method.lineno - 1, method.end_lineno


def apply_replacement(original, answer, compact=False):
    parsed_answer = json.loads(answer)
    start, end = method_span(original)
    lines = original.splitlines(keepends=True)
    if compact:
        method_lines = lines[start:end]
        next_start = len(method_lines) + 1
        for edit in sorted(parsed_answer["edits"], key=lambda change: change["start"], reverse=True):
            first, last = edit["start"], edit["end"]
            if not isinstance(first, int) or not isinstance(last, int) or not 1 <= first <= last < next_start:
                raise ValueError("Edit ranges must use valid, non-overlapping method line numbers")
            replacement = edit["replacement"].rstrip("\n")
            method_lines[first - 1:last] = [replacement + "\n"] if replacement else []
            next_start = first
        method = textwrap.dedent("".join(method_lines)).strip()
    else:
        method = textwrap.dedent(parsed_answer["replacement"]).strip()
    parsed = ast.parse(method)
    if len(parsed.body) != 1 or not isinstance(parsed.body[0], ast.FunctionDef) or parsed.body[0].name != "from_file":
        raise ValueError("Emit only the complete from_file method definition, without backticks")
    updated = "".join(lines[:start]) + textwrap.indent(method, "    ") + "\n" + "".join(lines[end:])
    ast.parse(updated)
    if updated == original:
        raise ValueError("The method was unchanged")
    return updated


def cpu_limit():
    resource.setrlimit(resource.RLIMIT_CPU, (20, 20))


def score(run, repo, python):
    temp = run / "pytest-tmp"
    temp.mkdir(exist_ok=True)
    runtime = Path(subprocess.check_output([str(python), "-B", "-c", "import sys; print(sys.base_prefix)"], text=True).strip())
    def path_rule(path):
        return f"(subpath {json.dumps(str(path))})"
    system_root = Path(Path.home().anchor)
    protected = sorted({Path.home().parent, system_root / "Volumes", system_root / "private"})
    protected_rules = " ".join(path_rule(path) for path in protected)
    profile = (
        '(version 1)(allow default)(deny network*)(deny process-fork)'
        f'(deny file-read-data (require-all (require-any {protected_rules}) '
        f'(require-not (require-any {path_rule(runtime)} {path_rule(python.parent.parent)} '
        f'{path_rule(repo)} {path_rule(temp)}))))'
        f'(deny file-write* (require-not (require-any {path_rule(temp)} (literal "/dev/null"))))'
    )
    command = ["/usr/bin/sandbox-exec", "-p", profile, str(python), "-B", "-m", "pytest",
               "-q", "-p", "no:cacheprovider", "tests/test_config.py"]
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(repo / "src"), "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
           "TMPDIR": str(temp), "XDG_CACHE_HOME": str(temp)}
    completed = subprocess.run(command, cwd=repo, env=env, timeout=40, capture_output=True, text=True,
                               check=False, preexec_fn=cpu_limit)
    return {"exit": completed.returncode, "output": (completed.stdout + completed.stderr)[-4500:]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkout", type=Path, required=True, help="Archive or checkout of the pinned Flask base commit")
    parser.add_argument("--test-patch", type=Path, required=True, help="Original test_patch for pallets__flask-4992 from SWE-bench Lite")
    parser.add_argument("--python", type=Path, required=True, help="Python executable in the dedicated scoring environment")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--prefill-json", action="store_true")
    parser.add_argument("--budget", type=int, default=0)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--input-target", type=int, default=0)
    parser.add_argument("--compact-edit", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()
    if args.attempts < 1 or args.budget < 0 or args.input_target < 0:
        parser.error("Attempts must be positive. Budget and input target must be non-negative.")
    checkout = args.checkout.resolve()
    run = args.output.resolve()
    if run.is_relative_to(checkout):
        parser.error("Output must be outside the input checkout")
    python = args.python.expanduser().absolute()
    test_patch = args.test_patch.resolve()
    if hashlib.sha256((checkout / SOURCE).read_bytes()).hexdigest() != SOURCE_SHA256:
        parser.error("Flask config.py differs from the pinned base version")
    if hashlib.sha256(test_patch.read_bytes()).hexdigest() != TEST_PATCH_SHA256:
        parser.error("Test patch differs from the official SWE-bench Lite fixture")
    run.mkdir(mode=0o700)
    repo = run / "repo"
    shutil.copytree(checkout, repo, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".git"))
    subprocess.run(["git", "apply", str(test_patch)], cwd=repo, check=True)
    baseline = score(run, repo, python)
    save(run / "baseline.json", baseline)
    if baseline["exit"] != 1 or "unexpected keyword argument 'text'" not in baseline["output"]:
        raise RuntimeError(f"The official baseline did not fail at the expected API behavior: {baseline}")
    if args.baseline_only:
        print(baseline["output"])
        return 0

    source = repo / SOURCE
    original = source.read_text(encoding="utf-8")
    problem = PROBLEM
    start, end = method_span(original)
    excerpt = "".join(original.splitlines(keepends=True)[start:end])
    system = "You make a focused Python change. "
    if args.compact_edit:
        system += "Return an edits array. Each edit has integer start and end for inclusive method line numbers, and a replacement string for those lines. Preserve indentation and edit only the lines needed. Do not repeat the original code."
        problem += "\nThe public API uses a boolean keyword `text` with default True. `text=False` opens the file in binary mode."
        excerpt = "\n".join(f"{i}: {line}" for i, line in enumerate(excerpt.splitlines(), 1))
    else:
        system += "Return a JSON object with one replacement string containing the complete updated from_file method."
    system += " Preserve the rest of the source. A controller applies your edit and checks the unmodified tests."
    user = f"{problem.strip()}\n\nHere is the original method:\n```python\n{excerpt.rstrip()}\n```"
    if args.input_target:
        base = f"http://127.0.0.1:{args.port}"
        context_files = ("app.py", "helpers.py", "ctx.py", "blueprints.py", "testing.py")
        surrounding = "\n\n".join(f"# src/flask/{name}\n{(repo / 'src/flask' / name).read_text()}" for name in context_files)
        reserved = len(query(base, "/tokenize", {"content": system + user}, timeout=40)["tokens"]) + 70
        if args.input_target <= reserved:
            raise ValueError("Input target leaves no room for surrounding code")
        available = query(base, "/tokenize", {"content": surrounding}, timeout=40)["tokens"]
        excerpt_ids = available[:args.input_target - reserved]
        surrounding = query(base, "/detokenize", {"tokens": excerpt_ids}, timeout=40)["content"]
        user = f"Here are surrounding files from the same Flask checkout for context. The edit target follows.\n\n{surrounding}\n\n{user}"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    bodies = []
    started = time.monotonic()
    passed = False
    for attempt in range(1, args.attempts + 1):
        history = messages + ([{"role": "assistant", "content": "{"}] if args.prefill_json else [])
        body = {"model": "inkling-2-mini", "messages": history, "stream": True,
                "stream_options": {"include_usage": True}, "temperature": 0, "max_tokens": 768,
                "cache_prompt": attempt > 1, "reasoning_budget_tokens": args.budget,
                "chat_template_kwargs": {"enable_thinking": args.budget > 0},
                "response_format": {"type": "json_schema", "json_schema": {
                    "name": "edit", "strict": True, "schema": EDIT_SCHEMA if args.compact_edit else SCHEMA}}}
        if args.prefill_json:
            body["echo"] = True
        save(run / f"request-{attempt}.json", body)
        metric, answer, calls = request(args.port, body, run / f"response-{attempt}.stream")
        try:
            if calls:
                raise ValueError("Unexpected tool call")
            replacement = apply_replacement(original, answer, args.compact_edit)
        except (ValueError, TypeError, SyntaxError, KeyError) as error:
            metric["patch_error"] = str(error)
            feedback = "Patch could not be applied: " + str(error)
        else:
            source.write_text(replacement, encoding="utf-8")
            metric["tests"] = score(run, repo, python)
            passed = metric["tests"]["exit"] == 0
            feedback = metric["tests"]["output"]
        bodies.append(metric)
        save(run / f"metrics-{attempt}.json", metric)
        if passed:
            break
        source.write_text(original, encoding="utf-8")
        messages.extend([{"role": "assistant", "content": answer}, {"role": "user", "content": f"The controller restored the original method after this check. Return a revised replacement.\n\n{feedback[-1500:]}"}])

    elapsed = time.monotonic() - started
    diff = "".join(difflib.unified_diff(original.splitlines(keepends=True), source.read_text().splitlines(keepends=True),
                                       fromfile=f"a/{SOURCE}", tofile=f"b/{SOURCE}"))
    (run / "candidate.diff").write_text(diff, encoding="utf-8")
    summary = {"case": "pallets__flask-4992", "official_test_patch": True, "guided_method_edit": True,
               "compact_edit": args.compact_edit, "reasoning_budget": args.budget, "input_target": args.input_target,
               "assistant_prefill": args.prefill_json, "passed": passed, "attempts": len(bodies), "wall_s": elapsed,
               "requests": bodies}
    save(run / "summary.json", summary)
    print({k: v for k, v in summary.items() if k != "requests"}, flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
