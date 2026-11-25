#!/usr/bin/env python3
"""Deterministic loopback-only OpenAI chat-completions fixture for local tests.

It is not started by importing this module. The two fixed replies call the
mini-swe-agent ``bash`` tool to create a proof file, then submit the task.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


FIRST_COMMAND = "printf fixture-ok > mini_swe_fixture.txt"
SUBMIT_COMMAND = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


def completion(model: str, request_number: int) -> dict[str, Any]:
    command = FIRST_COMMAND if request_number == 1 else SUBMIT_COMMAND
    content = "Create the fixture proof file." if request_number == 1 else "Submit the completed fixture task."
    return {
        "id": f"chatcmpl-fixture-{request_number}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content,
                "tool_calls": [{
                    "id": f"call_fixture_{request_number}",
                    "type": "function",
                    "function": {"name": "bash", "arguments": json.dumps({"command": command})},
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


class FixtureServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], handler: type[BaseHTTPRequestHandler], log_path: Path) -> None:
        super().__init__(address, handler)
        self.log_path = log_path
        self.request_number = 0
        self.request_lock = threading.Lock()

    def next_response(self, model: str) -> tuple[int, dict[str, Any]]:
        with self.request_lock:
            self.request_number += 1
            number = self.request_number
        return number, completion(model, number)

    def record(self, event: dict[str, Any]) -> None:
        with self.log_path.open("a") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")


class Handler(BaseHTTPRequestHandler):
    server: FixtureServer

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        if self.path == "/health":
            self.server.record({"event": "health"})
            self.send_json(HTTPStatus.OK, {"status": "ok"})
            return
        self.send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found", "type": "not_found"}})

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found", "type": "not_found"}})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            model = payload["model"]
            if not isinstance(model, str) or not model:
                raise ValueError("model must be a non-empty string")
        except (ValueError, json.JSONDecodeError, KeyError):
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": {"message": "invalid OpenAI request body", "type": "invalid_request_error"}})
            return
        number, response = self.server.next_response(model)
        self.server.record({"event": "chat_completion", "request_number": number})
        self.send_json(HTTPStatus.OK, response)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--log", type=Path, default=Path(__file__).with_name("openai-fixture.log"))
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be in 1..65535")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    server = FixtureServer(("127.0.0.1", args.port), Handler, args.log)
    server.record({"event": "started", "bind": "127.0.0.1", "port": args.port})
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.record({"event": "stopped"})
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
