#!/usr/bin/env python3
"""Mock-only cancellation tests; no Pier sandbox, model, or Docker is started."""
from __future__ import annotations

import asyncio
import importlib.util
import sys
import tempfile
import threading
import types
from pathlib import Path


ROOT = Path(__file__).parents[1]


def install_base_stubs() -> None:
    for name in ("pier", "pier.agents", "pier.environments", "pier.models", "pier.models.agent",
                 "pier.agents.installed", "pier.utils"):
        sys.modules.setdefault(name, types.ModuleType(name))
    agents = types.ModuleType("pier.agents.base")
    agents.BaseAgent = object
    environments = types.ModuleType("pier.environments.base")
    environments.BaseEnvironment = object
    context = types.ModuleType("pier.models.agent.context")
    context.AgentContext = object
    yaml = types.ModuleType("yaml")
    yaml.safe_load = lambda _: {}
    sys.modules.update({"pier.agents.base": agents, "pier.environments.base": environments,
                        "pier.models.agent.context": context, "yaml": yaml})


def load_host():
    install_base_stubs()
    spec = importlib.util.spec_from_file_location("deep_host_mini_swe_cancel", ROOT / "deep_host_mini_swe.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class BlockingModel:
    def __init__(self, started: threading.Event, release: threading.Event) -> None:
        self.started, self.release, self.calls = started, release, 0

    def query(self, _: list[dict]) -> dict:
        self.calls += 1
        self.started.set()
        assert self.release.wait(2), "test did not release the simulated HTTP request"
        return {"role": "assistant", "content": "late",
                "extra": {"actions": [{"command": "must-not-run"}]}}


class FastModel:
    def __init__(self) -> None:
        self.calls = 0

    def query(self, _: list[dict]) -> dict:
        self.calls += 1
        return {"role": "assistant", "content": "ok", "extra": {"actions": [{"command": "echo ok"}]}}


class Result:
    return_code, stdout, stderr = 0, "ok\n", ""


class AsyncEnvironment:
    def __init__(self) -> None:
        self.commands: list[str] = []

    async def exec(self, *, command: str, **_: object) -> Result:
        self.commands.append(command)
        return Result()


class Logger:
    def info(self, *_: object) -> None:
        pass

    def exception(self, *_: object) -> None:
        pass


def install_agent_stubs(completed: threading.Event, captured: dict[str, object]) -> None:
    mini = types.ModuleType("minisweagent")
    mini.__path__, mini.__version__ = [], "test"
    agents_pkg = types.ModuleType("minisweagent.agents")
    agents_pkg.__path__ = []
    default = types.ModuleType("minisweagent.agents.default")
    exceptions = types.ModuleType("minisweagent.exceptions")
    exceptions.Submitted = type("Submitted", (Exception,), {})

    class FakeDefaultAgent:
        def __init__(self, model, env, **_: object) -> None:
            self.model, self.env, self.n_calls, self.cost = model, env, 0, 0.0
            captured["model"] = model

        def run(self, _: str) -> dict:
            try:
                self.n_calls += 1
                message = self.model.query([])
                for action in message.get("extra", {}).get("actions", []):
                    self.env.execute(action)
                self.n_calls += 1
                self.model.query([])  # would be the next agent iteration
            finally:
                completed.set()
            return {"exit_status": "done"}

    default.DefaultAgent = FakeDefaultAgent
    converter = types.ModuleType("pier.agents.installed.mini_swe_agent")
    converter.convert_and_save_trajectory = lambda *_: None
    metrics = types.ModuleType("pier.utils.trajectory_metrics")
    metrics.populate_context_from_final_metrics = lambda *_: None
    sys.modules.update({"minisweagent": mini, "minisweagent.agents": agents_pkg,
                        "minisweagent.agents.default": default, "minisweagent.exceptions": exceptions,
                        "pier.agents.installed.mini_swe_agent": converter,
                        "pier.utils.trajectory_metrics": metrics})


def make_agent(host, logs_dir: Path, model, events: list[str]):
    agent = object.__new__(host.DeepSWEHostMiniSweAgent)
    agent._config_file = "mock"
    agent._config_yaml = None
    agent._api_base, agent._api_key, agent._model_class, agent._model_kwargs = None, "local", "litellm", {}
    agent._step_limit = agent._cost_limit = agent._command_timeout = agent._deterministic_outputs_file = None
    agent.logs_dir, agent.logger, agent.model_name = logs_dir, Logger(), "mock"
    agent._load_config = lambda: {"agent": {}, "environment": {"env": {}}}
    agent._build_model = lambda _: model

    async def detect(_: object) -> str:
        return "/app"

    async def capture(_: object, __: str, *, base_head=None):
        events.append("capture_after" if base_head else "capture_before")
        return {"head": "a" * 40}

    async def commit(_: object, __: str) -> None:
        events.append("commit")

    agent._detect_cwd, agent._capture_git_evidence, agent._commit_agent_work = detect, capture, commit
    return agent


async def main() -> None:
    host = load_host()

    # Cancellation during a blocked query discards the late response.  The
    # worker exits and neither the action nor the next query occurs.
    cancelled, started, release = threading.Event(), threading.Event(), threading.Event()
    blocked = BlockingModel(started, release)
    guarded = host.CancellationAwareModel(blocked, cancelled)
    commands: list[str] = []

    def worker() -> str:
        try:
            message = guarded.query([])
            commands.extend(action["command"] for action in message["extra"]["actions"])
            guarded.query([])
        except host.WorkerCancelled:
            return "cancelled"
        return "unexpected"

    task = asyncio.create_task(asyncio.to_thread(worker))
    assert await asyncio.to_thread(started.wait, 1)
    cancelled.set()
    release.set()
    assert await task == "cancelled"
    assert blocked.calls == 1 and commands == []

    # Normal wrappers retain query-to-command behavior, while a pre-cancelled
    # sandbox call is never scheduled.
    install_agent_stubs(threading.Event(), {})
    env, normal_cancelled = AsyncEnvironment(), threading.Event()
    sandbox = host.PierSandboxEnvironment(env, asyncio.get_running_loop(), cwd="/app", env={}, timeout=1,
                                           cancelled=normal_cancelled)
    fast = FastModel()
    message = await asyncio.to_thread(host.CancellationAwareModel(fast, normal_cancelled).query, [])
    output = await asyncio.to_thread(sandbox.execute, message["extra"]["actions"][0])
    assert output["returncode"] == 0 and fast.calls == 1 and env.commands == ["echo ok"]
    normal_cancelled.set()
    try:
        await asyncio.to_thread(sandbox.execute, {"command": "must-not-run"})
    except host.WorkerCancelled:
        pass
    else:
        raise AssertionError("cancelled sandbox command was scheduled")
    assert env.commands == ["echo ok"]

    # This exercises the real async adapter boundary with a blocked mock
    # model.  It must re-raise cancellation, signal the thread-safe event, and
    # skip commit/evidence writes that would submit a cancelled trial.
    completed, captured, events = threading.Event(), {}, []
    install_agent_stubs(completed, captured)
    actual_started, actual_release = threading.Event(), threading.Event()
    actual_model = BlockingModel(actual_started, actual_release)
    with tempfile.TemporaryDirectory() as directory:
        logs = Path(directory) / "logs"
        agent = make_agent(host, logs, actual_model, events)
        run_task = asyncio.create_task(agent.run("task", AsyncEnvironment(), types.SimpleNamespace(metadata={})))
        assert await asyncio.to_thread(actual_started.wait, 1)
        run_task.cancel()
        try:
            await run_task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("adapter swallowed cancellation")
        assert captured["model"]._cancelled.is_set()
        actual_release.set()
        assert await asyncio.to_thread(completed.wait, 1), "worker did not exit after query returned"
        assert actual_model.calls == 1 and events == ["capture_before"]
        assert not (logs / "result.json").exists()
        assert not (logs / "deep-swe-git-evidence.json").exists()

    # A normal adapter run still executes the command and performs the commit
    # flow after the worker has completed.
    completed, captured, events = threading.Event(), {}, []
    install_agent_stubs(completed, captured)
    with tempfile.TemporaryDirectory() as directory:
        logs, env = Path(directory) / "logs", AsyncEnvironment()
        agent = make_agent(host, logs, FastModel(), events)
        await agent.run("task", env, types.SimpleNamespace(metadata={}))
        assert env.commands == ["echo ok"]
        assert events == ["capture_before", "commit", "capture_after"]
        assert (logs / "result.json").exists()

    print("cancellation mocks: ok")


asyncio.run(main())
