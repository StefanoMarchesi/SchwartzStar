#!/usr/bin/env python3
"""Run a synthetic coding-agent probe against an existing loopback llama-server."""
import argparse
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check import snapshot

READ_PROMPT = (
    "Use a tool to read marker.txt in this repository. "
    "Return exactly the entire value from the file. Do not modify any file."
)
FIX_PROMPT = (
    "Fix the bug in calc.py so clamp(x, lo, hi) returns the nearest bound "
    "when x is outside the inclusive interval. Edit only calc.py. "
    "Run python3 -B -m unittest -v using a tool and report the result."
)
COMPACT_CODEX_PROMPT = (
    "You are a coding agent in a synthetic test repository. Use exec_command to read files and run tests. "
    "For edits use the available tools, and respect the user instructions and the sandbox. "
    "Keep commands and final responses short. Do not use the network or access files outside this repository.\n"
)
FIXTURE = {
    "calc.py": "def clamp(x, lo, hi):\n    return min(max(x, lo), hi + 1)\n",
    "test_calc.py": """import unittest
from calc import clamp


class ClampTest(unittest.TestCase):
    def test_inclusive_bounds(self):
        self.assertEqual([clamp(x, 2, 7) for x in (0, 2, 4, 7, 10)], [2, 2, 4, 7, 7])
""",
}


def create_fixture(root):
    root.mkdir()
    marker = f"INKLING-E2E-{uuid.uuid4().hex}"
    for name, value in (FIXTURE | {"marker.txt": marker + "\n"}).items():
        (root / name).write_text(value)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "--", *FIXTURE, "marker.txt"], check=True)
    original = subprocess.run(
        [sys.executable, "-B", "-m", "unittest", "-v"],
        cwd=root, capture_output=True, text=True,
    )
    if original.returncode != 1 or "FAIL: test_inclusive_bounds" not in original.stderr:
        raise RuntimeError(f"Fixture did not fail as intended: {original.stderr}")
    return marker, original.stderr


def audit_request(payload):
    items = payload.get("input", payload.get("messages", []))
    return {
        "input_items": len(items),
        "instructions_chars": len(payload.get("instructions") or ""),
        "tool_names": [t.get("name") or t.get("function", {}).get("name") for t in payload.get("tools", [])],
        "previous_response_id": bool(payload.get("previous_response_id")),
        "tool_result_ids": [i.get("tool_call_id") or i.get("call_id") for i in items
                            if i.get("role") == "tool" or i.get("type") == "function_call_output"],
    }


class TraceServer(ThreadingHTTPServer):
    def __init__(self, directory, upstream_port, pid, footprint):
        super().__init__(("127.0.0.1", 0), TraceHandler)
        self.directory = directory
        self.upstream_port = upstream_port
        self.pid = pid
        self.footprint = footprint
        self.requests = []

    def peak_mib(self):
        data = subprocess.check_output([self.footprint, str(self.pid)], text=True)
        return round(int(data) / 1048576, 1)


class TraceHandler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        self.forward()

    def do_POST(self):
        self.forward()

    def forward(self):
        if self.path.split("?", 1)[0] not in ("/v1/responses", "/v1/chat/completions", "/v1/models", "/v1/responses/compact"):
            self.send_error(404)
            return
        server = self.server
        started = time.monotonic()
        data = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        entry = {
            "path": self.path, "start_s": started - server.turn_started,
            "first_text_s": None, "first_tool_s": None, "usage": None, "timings": None,
            "tool_call_ids": [],
        }
        ordinal = len(server.requests) + 1
        server.requests.append(entry)
        stem = server.directory / f"request-{ordinal:02}"
        if data:
            stem.with_suffix(".json").write_bytes(data)
            entry.update(audit_request(json.loads(data)))
        print(f"HTTP {ordinal} {self.path} {len(data)} bytes, {entry.get('input_items', 0)} items", flush=True)
        connection = http.client.HTTPConnection("127.0.0.1", server.upstream_port, timeout=3600)
        try:
            connection.request(self.command, self.path, body=data or None, headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            entry["status"] = response.status
            content_type = response.getheader("Content-Type", "application/json")
            self.send_response(response.status)
            self.send_header("Content-Type", content_type)
            self.end_headers()
            with stem.with_suffix(".stream").open("wb") as wire:
                if "text/event-stream" in content_type:
                    for line in iter(response.readline, b""):
                        now = time.monotonic() - started
                        wire.write(line)
                        wire.flush()
                        if line.startswith(b"data:") and line[5:].strip() != b"[DONE]":
                            self.observe(entry, json.loads(line[5:]), now)
                        self.wfile.write(line)
                        self.wfile.flush()
                else:
                    body = response.read()
                    wire.write(body)
                    if response.status >= 400:
                        entry["error"] = json.loads(body) if "json" in content_type else body.decode(errors="replace")
                    self.wfile.write(body)
        except (OSError, http.client.HTTPException) as error:
            entry["transport_error"] = str(error)
        finally:
            connection.close()
            entry["total_s"] = time.monotonic() - started
            entry["physical_footprint_mib"] = server.peak_mib()
            print(f"HTTP {ordinal} ended: {entry.get('status')}, {entry['total_s']:.2f}s", flush=True)

    @staticmethod
    def observe(entry, event, elapsed):
        kind = event.get("type")
        response = event.get("response", {})
        entry["usage"] = response.get("usage") or event.get("usage") or entry["usage"]
        entry["timings"] = event.get("timings") or response.get("timings") or entry["timings"]
        item = event.get("item", {})
        choices = event.get("choices") or []
        delta = choices[0].get("delta", {}) if choices else {}
        call_ids = [call.get("id") for call in delta.get("tool_calls", [])]
        if item.get("type") == "function_call":
            call_ids.append(item.get("call_id"))
        entry["tool_call_ids"].extend(call_id for call_id in call_ids
                                     if call_id and call_id not in entry["tool_call_ids"])
        if (kind == "response.output_text.delta" and event.get("delta")) or delta.get("content"):
            if entry["first_text_s"] is None:
                entry["first_text_s"] = elapsed
        if (
            kind in ("response.output_item.added", "response.output_item.done") and item.get("type") == "function_call"
            or kind == "response.function_call_arguments.delta"
            or delta.get("tool_calls")
        ):
            if entry["first_tool_s"] is None:
                entry["first_tool_s"] = elapsed


def codex_cmd(codex, root, base_url, turn, instructions=None):
    provider = (
        'model_providers.inkling={name="Inkling local", base_url=' + json.dumps(base_url) +
        ', wire_api="responses", requires_openai_auth=false, supports_websockets=false, '
        'request_max_retries=0, stream_max_retries=0, stream_idle_timeout_ms=3600000}'
    )
    command = [
        codex, "-a", "never", "exec", "--ignore-user-config", "--ignore-rules", "--ephemeral",
        "--json", "--color", "never", "-C", str(root), "-m", "inkling-2-mini", "--sandbox",
        "read-only" if turn == "read" else "workspace-write",
        "--disable", "plugins", "--enable", "skip_host_skill_discovery",
        "-c", 'web_search="disabled"', "-c", 'model_provider="inkling"', "-c", provider,
        "-c", 'model_reasoning_effort="none"', "-c", "project_doc_max_bytes=0",
    ]
    if instructions:
        command.extend(["--disable", "multi_agent", "--disable", "goals", "--disable", "apps", "--disable", "view_image",
                        "-c", "skills.include_instructions=false", "-c", "allow_login_shell=false",
                        "-c", "tools.experimental_request_user_input.enabled=false",
                        "-c", f"model_instructions_file={json.dumps(str(instructions))}"])
    return command + [READ_PROMPT if turn == "read" else FIX_PROMPT]


def opencode_config(base_url, context):
    return {
        "provider": {
            "inkling": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Inkling local",
                "options": {"baseURL": base_url, "apiKey": "local", "timeout": 3600000, "includeUsage": True},
                "models": {"inkling-2-mini": {"name": "Inkling 2 Mini", "limit": {"context": context, "output": 512}}},
            },
        },
        "enabled_providers": ["inkling"],
        "autoupdate": False,
        "share": "disabled",
        "agent": {
            "inkling-read": {
                "mode": "primary", "description": "Read synthetic fixture",
                "prompt": "Use tools for the requested local task. Follow the user instructions exactly.",
                "permission": {"*": "deny", "read": "allow"},
            },
            "inkling-fix": {
                "mode": "primary", "description": "Repair synthetic fixture",
                "prompt": "Use tools for the requested local task. Follow the user instructions exactly.",
                "permission": {"*": "deny", "read": "allow", "edit": {"*": "deny", "calc.py": "allow"},
                               "bash": {"*": "deny", "python3 -B -m unittest -v": "allow"}},
            },
        },
    }


def opencode_cmd(root, turn):
    return [
        "opencode", "run", "--pure", "--dir", str(root), "--format", "json", "--agent", f"inkling-{turn}",
        "--title", f"Inkling synthetic fixture: {turn}",
        "-m", "inkling/inkling-2-mini", READ_PROMPT if turn == "read" else FIX_PROMPT,
    ]


def cli_evidence(path):
    messages, test_commands = [], []
    for line in path.read_text().splitlines():
        event = json.loads(line)
        if event.get("type") == "item.completed":
            item = event.get("item", {})
            if item.get("type") == "agent_message":
                messages.append(item.get("text", ""))
            if item.get("type") == "command_execution" and item.get("exit_code") == 0:
                test_commands.append(item.get("command", ""))
        elif event.get("type") == "text":
            messages.append(event.get("part", {}).get("text", ""))
        elif event.get("type") == "tool_use":
            part = event.get("part", {})
            state = part.get("state", {})
            if part.get("tool") == "bash" and state.get("status") == "completed":
                test_commands.append(state.get("input", {}).get("command", ""))
    return messages, any("unittest" in command for command in test_commands)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="New output directory for private wire evidence")
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--footprint", type=Path, required=True)
    parser.add_argument("--upstream-port", type=int, default=18080)
    parser.add_argument("--turn", choices=("read", "fix", "both"), default="both")
    parser.add_argument("--codex", default="codex")
    parser.add_argument("--isolated", action="store_true", help="Use disposable Codex state without account configuration")
    parser.add_argument("--compact-codex", action="store_true", help="Use a short Codex prompt and omit the skill catalog")
    parser.add_argument("--client", choices=("codex", "opencode"), default="codex")
    parser.add_argument("--context", type=int, default=4096, help="Context advertised to OpenCode")
    parser.add_argument("--max-footprint-mib", type=int, default=4500)
    opts = parser.parse_args()
    if opts.compact_codex and opts.client != "codex":
        parser.error("--compact-codex requires Codex")
    directory = opts.directory.resolve()
    directory.mkdir(mode=0o700, parents=False, exist_ok=False)
    root = directory / "fixture"
    marker, original = create_fixture(root)
    (directory / "test-before.txt").write_text(original)
    env = None
    if opts.isolated or opts.compact_codex:
        state = directory / "codex-state"
        state.mkdir()
        env = {k: os.environ[k] for k in ("PATH", "HOME", "TMPDIR", "SHELL", "LANG", "USER", "LOGNAME") if k in os.environ}
        env["CODEX_HOME"] = str(state)
    instructions = None
    if opts.compact_codex:
        instructions = directory / "instructions.md"
        instructions.write_text(COMPACT_CODEX_PROMPT)
    server = TraceServer(directory, opts.upstream_port, opts.pid, str(opts.footprint))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base_url = f"http://127.0.0.1:{server.server_port}/v1"
    if opts.client == "opencode":
        env = {k: os.environ[k] for k in ("PATH", "HOME", "TMPDIR", "SHELL", "LANG", "USER", "LOGNAME") if k in os.environ}
        env.update({
            "XDG_CONFIG_HOME": str(directory / "opencode-config"),
            "XDG_DATA_HOME": str(directory / "opencode-data"),
            "XDG_STATE_HOME": str(directory / "opencode-state"),
            "OPENCODE_CONFIG_CONTENT": json.dumps(opencode_config(base_url, opts.context)),
            "OPENCODE_DISABLE_MODELS_FETCH": "true", "OPENCODE_DISABLE_AUTOUPDATE": "true",
            "OPENCODE_DISABLE_EXTERNAL_SKILLS": "true", "OPENCODE_DISABLE_LSP_DOWNLOAD": "true",
            "OPENCODE_DISABLE_CLAUDE_CODE": "true",
        })
    print(f"Evidence: {directory}", flush=True)
    failed = False
    try:
        for turn in ("read", "fix") if opts.turn == "both" else (opts.turn,):
            before = snapshot()
            first = len(server.requests)
            server.turn_started = time.monotonic()
            command = (codex_cmd(opts.codex, root, base_url, turn, instructions) if opts.client == "codex"
                       else opencode_cmd(root, turn))
            with (directory / f"{turn}.jsonl").open("w") as events, (directory / f"{turn}.stderr").open("w") as errors:
                run = subprocess.run(command, cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=events, stderr=errors)
            elapsed = time.monotonic() - server.turn_started
            requests = server.requests[first:]
            call_ids = {call_id for request in requests for call_id in request["tool_call_ids"]}
            result_ids = {call_id for request in requests for call_id in request.get("tool_result_ids", [])}
            messages, test_used = cli_evidence(directory / f"{turn}.jsonl")
            report = dict(
                turn=turn, returncode=run.returncode, duration_s=elapsed,
                before=before, after=snapshot(), peak_physical_footprint_mib=server.peak_mib(),
                requests=requests, structured_tool_results=len(call_ids & result_ids),
            )
            patch = subprocess.check_output(["git", "-C", str(root), "diff", "--"], text=True)
            (directory / f"{turn}.diff").write_text(patch)
            changed = subprocess.check_output(["git", "-C", str(root), "diff", "--name-only"], text=True).splitlines()
            untracked = subprocess.check_output(["git", "-C", str(root), "ls-files", "--others", "--exclude-standard"], text=True).splitlines()
            report["fixture_changes"] = sorted(set(changed + untracked))
            if turn == "fix":
                test = subprocess.run([sys.executable, "-B", "-m", "unittest", "-v"], cwd=root, capture_output=True, text=True)
                (directory / "test-after.txt").write_text(test.stdout + test.stderr)
                report["independent_test_exit"] = test.returncode
                report["agent_test_command"] = test_used
                correct = test_used and test.returncode == 0 and report["fixture_changes"] == ["calc.py"]
            else:
                report["marker_in_final_message"] = bool(messages and messages[-1].strip() == marker)
                correct = report["marker_in_final_message"] and not report["fixture_changes"]
            report["passed"] = (correct and run.returncode == 0 and report["structured_tool_results"] > 0
                                and report["peak_physical_footprint_mib"] <= opts.max_footprint_mib)
            (directory / f"{turn}.summary.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2), flush=True)
            if report["peak_physical_footprint_mib"] > opts.max_footprint_mib or run.returncode:
                failed = True
                break
            failed |= not report["passed"]
    finally:
        server.shutdown()
        server.server_close()
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
