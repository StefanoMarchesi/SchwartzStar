#!/usr/bin/env python3
"""Run two tool-assisted PHP turns against a loopback Chat Completions server."""
import argparse
import http.client
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from agent_probe import TraceHandler, snapshot
from php_fixture import SHORT_CONTEXT, TURN_1, TURN_2, populate

FILES = ("src/invoice.php", "src/costs.php")
SYSTEM = (
    "You are a coding agent working in a synthetic PHP repository. "
    "Use the provided tools to read files, edit both source files, and run the test. "
    "Use tools for all edits and tests. Keep your final response brief."
)
STRING = {"type": "string"}
PATH = {"type": "string", "enum": list(FILES)}


def tool(name, description, properties):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties,
                       "required": list(properties), "additionalProperties": False},
    }}


TOOLS = [
    tool("read_file", "Read a complete PHP source file", {"path": PATH}),
    tool("replace_text", "Replace exactly one occurrence of old text in a source file",
         {"path": PATH, "old": STRING, "new": STRING}),
    tool("run_tests", "Run the PHP test for the current turn and return its output", {}),
]


def request(port, body, target):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3600)
    started = time.monotonic()
    metric = dict(first_text_s=None, first_tool_s=None, usage=None, timings=None, tool_call_ids=[])
    text, calls = [], {}
    try:
        connection.request("POST", "/v1/chat/completions", json.dumps(body),
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        metric["http"] = response.status
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {response.read().decode(errors='replace')}")
        with target.open("wb") as trace:
            for line in iter(response.readline, b""):
                trace.write(line)
                if not line.startswith(b"data:"):
                    continue
                value = line[5:].strip()
                if value == b"[DONE]":
                    break
                packet = json.loads(value)
                if packet.get("error"):
                    raise RuntimeError(packet["error"])
                TraceHandler.observe(metric, packet, time.monotonic() - started)
                for choice in packet.get("choices") or []:
                    delta = choice.get("delta", {})
                    text.append(delta.get("content") or "")
                    for data in delta.get("tool_calls", []):
                        part = calls.setdefault(data["index"], {"id": "", "type": "function",
                                                                 "function": {"name": "", "arguments": ""}})
                        part["id"] += data.get("id") or ""
                        function = data.get("function", {})
                        for key in ("name", "arguments"):
                            part["function"][key] += function.get(key) or ""
                    if choice.get("finish_reason"):
                        metric["finish_reason"] = choice["finish_reason"]
    finally:
        connection.close()
    metric["total_s"] = time.monotonic() - started
    return metric, "".join(text), [calls[index] for index in sorted(calls)]


def run_tests(php, root, turn):
    profile = "(version 1)(allow default)(deny network*)(deny file-write*)"
    disabled = "exec,passthru,shell_exec,system,proc_open,popen,pcntl_exec,dl,mail"
    result = subprocess.run(
        ["/usr/bin/sandbox-exec", "-p", profile, php, "-n", "-d", f"open_basedir={root}",
         "-d", f"disable_functions={disabled}", "tests/run.php", str(turn)],
        cwd=root, capture_output=True, text=True, timeout=10,
    )
    return {"exit": result.returncode, "output": (result.stdout + result.stderr)[:4096]}


def test_passed(result, turn):
    return result.get("exit") == 0 and result.get("output", "").strip() == f"turn {turn} ok"


def run_tool(call, root, php, turn):
    function = call["function"]
    name = function["name"]
    try:
        args = json.loads(function["arguments"])
        if name == "run_tests" and args == {}:
            return run_tests(php, root, turn)
        if name not in ("read_file", "replace_text") or args.get("path") not in FILES:
            return {"error": "Unknown tool or file"}
        path = root / args["path"]
        text = path.read_text()
        if name == "read_file":
            return {"path": args["path"], "content": text}
        old = args["old"]
        if not old or text.count(old) != 1:
            return {"error": "old must match exactly once. Read the file and retry."}
        path.write_text(text.replace(old, args["new"], 1))
        return {"modified": args["path"]}
    except (ValueError, KeyError, OSError, AttributeError, TypeError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New directory for the synthetic fixture and raw trace")
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--footprint", type=Path, required=True)
    parser.add_argument("--context-file", type=Path)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--reasoning-budget", type=int)
    parser.add_argument("--disable-thinking", action="store_true")
    opts = parser.parse_args()
    php = shutil.which("php")
    if not php:
        parser.error("php is required")
    opts.output.mkdir(mode=0o700)
    root = opts.output.resolve() / "fixture"
    root.mkdir()
    populate(root)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "--", *FILES, "tests/run.php"], check=True)
    baseline = run_tests(php, root, 1)
    (opts.output / "baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
    if baseline["exit"] != 1:
        raise RuntimeError(f"The initial test did not fail as expected: {baseline}")
    context = opts.context_file.read_text() if opts.context_file else SHORT_CONTEXT
    messages = [{"role": "system", "content": SYSTEM}]
    for turn, task in enumerate((TURN_1, TURN_2), 1):
        before = snapshot()
        started = time.monotonic()
        first_green = None
        requests = []
        initial_files = {name: (root / name).read_text() for name in FILES}
        content = context + "\n\n" + task if turn == 1 else task
        messages.append({"role": "user", "content": content})
        for step in range(1, opts.steps + 1):
            body = {"model": opts.model, "messages": messages, "tools": TOOLS, "stream": True,
                    "stream_options": {"include_usage": True}, "temperature": 0,
                    "max_tokens": opts.max_tokens}
            if turn == 1 and step == 1:
                body["cache_prompt"] = False
            if opts.reasoning_budget is not None:
                body["reasoning_budget_tokens"] = opts.reasoning_budget
            if opts.disable_thinking:
                body["chat_template_kwargs"] = {"enable_thinking": False}
            prefix = opts.output / f"turn-{turn}-request-{step:02}"
            prefix.with_suffix(".json").write_text(json.dumps(body, ensure_ascii=False) + "\n")
            offset = time.monotonic() - started
            print(f"Turn {turn}, request {step}", flush=True)
            metric, answer, calls = request(opts.port, body, prefix.with_suffix(".stream"))
            metric["start_s"] = offset
            requests.append(metric)
            prefix.with_suffix(".metrics.json").write_text(json.dumps(metric, indent=2) + "\n")
            messages.append({"role": "assistant", "content": answer or None, **({"tool_calls": calls} if calls else {})})
            for call in calls:
                result = run_tool(call, root, php, turn)
                if call["function"]["name"] == "run_tests" and test_passed(result, turn) and first_green is None:
                    first_green = time.monotonic() - started
                messages.append({"role": "tool", "tool_call_id": call["id"],
                                 "content": json.dumps(result, ensure_ascii=False)})
                print(f"  {call['function']['name']}: {result.get('exit', result.get('error', 'ok'))}", flush=True)
            if not calls:
                break
        verified = run_tests(php, root, turn)
        previous_ok = turn == 1 or test_passed(run_tests(php, root, 1), 1)
        green = test_passed(verified, turn) and previous_ok and first_green is not None
        diff = subprocess.check_output(["git", "-C", str(root), "diff", "--"], text=True)
        both_modified = all(initial_files[name] != (root / name).read_text() for name in FILES)
        (opts.output / f"turn-{turn}.diff").write_text(diff)
        peak = int(subprocess.check_output([str(opts.footprint), str(opts.pid)], text=True)) / 1048576
        rss = subprocess.check_output(["ps", "-o", "rss=", "-p", str(opts.pid)], text=True).strip()
        pressure = subprocess.check_output(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True).strip()
        elapsed = time.monotonic() - started
        report = dict(turn=turn, passed=green and both_modified, both_source_files_changed=both_modified,
                      test=verified, turn_1_regression_passed=previous_ok, time_to_green_s=first_green,
                      total_s=elapsed, requests=requests, footprint_mib=round(peak, 1),
                      rss_end_mib=int(rss) / 1024, vm_pressure_level=int(pressure),
                      before=before, after=snapshot())
        (opts.output / f"turn-{turn}.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Turn {turn}: {'PASS' if report['passed'] else 'FAIL'}, {elapsed:.1f} s", flush=True)
        if not report["passed"]:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
