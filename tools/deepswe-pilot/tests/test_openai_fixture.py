"""Offline serialization checks for the deterministic OpenAI fixture."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).parent
spec = importlib.util.spec_from_file_location("fixture", ROOT / "openai_fixture.py")
fixture = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(fixture)


def command_from(response: dict) -> str:
    choice = response["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    call = choice["message"]["tool_calls"][0]
    assert call["type"] == "function" and call["function"]["name"] == "bash"
    return json.loads(call["function"]["arguments"])["command"]


def main() -> None:
    first = fixture.completion("openai/fixture", 1)
    second = fixture.completion("openai/fixture", 2)
    assert command_from(first) == fixture.FIRST_COMMAND
    assert command_from(second) == fixture.SUBMIT_COMMAND
    assert first["usage"] == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    assert json.loads(json.dumps(first))["object"] == "chat.completion"
    print("OpenAI fixture serialization: ok")


if __name__ == "__main__":
    main()
