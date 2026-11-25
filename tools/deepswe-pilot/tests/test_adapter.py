"""Lightweight contract test; it never starts a container or model server."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from unittest.mock import patch
from pathlib import Path
import sys

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "true")
sys.path.insert(0, str(Path(__file__).parents[1]))

from deep_host_mini_swe import DeepSWEHostMiniSweAgent, PierSandboxEnvironment
from pier.environments.base import ExecResult
from pier.models.agent.context import AgentContext


class FakeEnvironment:
    def __init__(self, *, detected_root: str = "/app") -> None:
        self.commands: list[str] = []
        self.detected_root = detected_root

    async def exec(self, command: str, cwd: str | None = None, env=None, timeout_sec=None, user=None) -> ExecResult:
        self.commands.append(command)
        if command.startswith("for d in /app /testbed"):
            if self.detected_root in {"/app", "/testbed"}:
                return ExecResult(stdout=self.detected_root + "\n", stderr="", return_code=0)
            return ExecResult(stdout="", stderr="no supported git worktree", return_code=42)
        if "rev-parse --show-toplevel" in command:
            if self.detected_root in command:
                return ExecResult(stdout=self.detected_root + "\n", stderr="", return_code=0)
            return ExecResult(stdout="", stderr="not a git repository", return_code=128)
        if "rev-parse --verify HEAD" in command:
            return ExecResult(stdout="a" * 40 + "\n", stderr="", return_code=0)
        if "status --porcelain" in command:
            return ExecResult(stdout="", stderr="", return_code=0)
        if "diff --binary" in command:
            return ExecResult(stdout="diff --git a/example b/example\n", stderr="", return_code=0)
        if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in command:
            return ExecResult(stdout="COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n", stderr="", return_code=0)
        return ExecResult(stdout="", stderr="", return_code=0)


async def test_detect_cwd_uses_testbed_when_app_is_not_a_repo() -> None:
    environment = FakeEnvironment(detected_root="/testbed")
    with tempfile.TemporaryDirectory() as directory:
        agent = DeepSWEHostMiniSweAgent(logs_dir=Path(directory), model_name="openai/mock")
        assert await agent._detect_cwd(environment) == "/testbed"
    assert environment.commands[0].startswith("for d in /app /testbed")


async def test_detect_cwd_fails_without_a_repo() -> None:
    environment = FakeEnvironment(detected_root="")
    with tempfile.TemporaryDirectory() as directory:
        agent = DeepSWEHostMiniSweAgent(logs_dir=Path(directory), model_name="openai/mock")
        try:
            await agent._detect_cwd(environment)
        except RuntimeError as exc:
            assert "no supported git worktree" in str(exc)
        else:
            raise AssertionError("missing git root was accepted")


async def test_commit_rejects_a_different_git_root() -> None:
    environment = FakeEnvironment(detected_root="/testbed")
    with tempfile.TemporaryDirectory() as directory:
        agent = DeepSWEHostMiniSweAgent(logs_dir=Path(directory), model_name="openai/mock")
        try:
            await agent._commit_agent_work(environment, "/app")
        except RuntimeError as exc:
            assert "refusing Git operation" in str(exc)
        else:
            raise AssertionError("commit was allowed outside the detected worktree")
    assert not any("git config --global" in command for command in environment.commands)


async def main() -> None:
    await test_detect_cwd_uses_testbed_when_app_is_not_a_repo()
    await test_detect_cwd_fails_without_a_repo()
    await test_commit_rejects_a_different_git_root()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        outputs = root / "outputs.json"
        outputs.write_text(json.dumps([{"role": "assistant", "content": "", "extra": {
            "actions": [{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}]}}]))
        agent = DeepSWEHostMiniSweAgent(
            logs_dir=root / "logs", model_name="openai/mock", logger=logging.getLogger("test"),
            deterministic_outputs_file=str(outputs), step_limit=1,
        )
        environment = FakeEnvironment()
        template_vars = PierSandboxEnvironment(
            environment, asyncio.get_running_loop(), cwd="/app", env={}, timeout=60
        ).get_template_vars()
        assert template_vars["system"] == "Linux"
        assert template_vars["machine"] == ""
        await agent.run("solve the mock task", environment, AgentContext())
        assert any("git -c user.name='DeepSWE host agent'" in command for command in environment.commands)
        assert (root / "logs" / "mini-swe-agent.trajectory.json").exists()
        assert (root / "logs" / "trajectory.json").exists()
        evidence = json.loads((root / "logs" / "deep-swe-git-evidence.json").read_text())
        assert evidence["before"]["head"] == "a" * 40
        assert evidence["after"]["patch_file"] == "deep-swe-submission.patch"
        assert (root / "logs" / "deep-swe-submission.patch").read_text().startswith("diff --git")

    # ATIF is an optional post-processing artifact. A conversion failure must
    # leave the native trajectory and committed patch available to Pier.
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        outputs = root / "outputs.json"
        outputs.write_text(json.dumps([{"role": "assistant", "content": "", "extra": {
            "actions": [{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}]}}]))
        agent = DeepSWEHostMiniSweAgent(
            logs_dir=root / "logs", model_name="openai/mock", logger=logging.getLogger("test"),
            deterministic_outputs_file=str(outputs), step_limit=1,
        )
        environment = FakeEnvironment()
        with patch("pier.agents.installed.mini_swe_agent.convert_and_save_trajectory", side_effect=RuntimeError("bad atif")):
            await agent.run("solve the mock task", environment, AgentContext())
        assert any("git -c user.name='DeepSWE host agent'" in command for command in environment.commands)
        assert (root / "logs" / "mini-swe-agent.trajectory.json").exists()
        assert json.loads((root / "logs" / "atif-conversion-error.json").read_text())["exception"] == "bad atif"


if __name__ == "__main__":
    asyncio.run(main())
    print("adapter contract mocks: ok")
