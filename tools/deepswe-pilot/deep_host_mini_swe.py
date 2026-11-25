"""Host-side mini-swe-agent adapter for DeepSWE v1.1/Pier.

The model loop runs on the Pier host.  Only shell commands requested by the
agent run in the task container, so the task can keep ``network_mode =
\"no-network\"`` and the local OpenAI-compatible endpoint is never exposed to
the container.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import threading
import uuid
from pathlib import Path
from typing import Any

import yaml

from pier.agents.base import BaseAgent
from pier.environments.base import BaseEnvironment
from pier.models.agent.context import AgentContext

_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
_SUPPORTED_GIT_ROOTS = ("/app", "/testbed")


class WorkerCancelled(RuntimeError):
    """Stop the synchronous agent loop after its async Pier trial is cancelled.

    ``asyncio.to_thread`` cannot stop a Python thread or a blocking model HTTP
    request.  The worker therefore cooperates at safe boundaries: before and
    after a model query, and before scheduling a sandbox command.
    """


class CancellationAwareModel:
    """Keep a cancelled worker from advancing after a blocked model query."""

    def __init__(self, model: Any, cancelled: threading.Event) -> None:
        self._model = model
        self._cancelled = cancelled

    def query(self, messages: list[dict]) -> dict:
        if self._cancelled.is_set():
            raise WorkerCancelled("Pier trial was cancelled before a model query")
        message = self._model.query(messages)
        # A response received after cancellation must not trigger an action or
        # a subsequent model call.
        if self._cancelled.is_set():
            raise WorkerCancelled("Pier trial was cancelled during a model query")
        return message

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)


class PierSandboxEnvironment:
    """Synchronous mini-swe-agent environment backed by async Pier exec."""

    def __init__(
        self, environment: BaseEnvironment, loop: asyncio.AbstractEventLoop, *, cwd: str,
        env: dict[str, str], timeout: int, cancelled: threading.Event | None = None,
    ) -> None:
        self._environment = environment
        self._loop = loop
        self.cwd = cwd
        self.env = dict(env)
        self.timeout = timeout
        self._cancelled = cancelled or threading.Event()

    def execute(self, action: dict[str, Any], cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        from minisweagent.exceptions import Submitted

        if self._cancelled.is_set():
            raise WorkerCancelled("Pier trial was cancelled before a sandbox command")
        command = action.get("command", "")
        command_timeout = timeout or self.timeout
        future = asyncio.run_coroutine_threadsafe(
            self._environment.exec(
                command=command,
                cwd=cwd or self.cwd,
                env=self.env or None,
                timeout_sec=command_timeout,
            ),
            self._loop,
        )
        try:
            result = future.result(timeout=command_timeout + 120)
            output: dict[str, Any] = {
                "output": (result.stdout or "") + (result.stderr or ""),
                "returncode": result.return_code,
                "exception_info": "",
            }
        except Exception as exc:
            output = {
                "output": "",
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {exc}",
                "extra": {"exception_type": type(exc).__name__, "exception": str(exc)},
            }

        lines = output["output"].lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == _MARKER and output["returncode"] == 0:
            submission = "".join(lines[1:])
            raise Submitted({"role": "exit", "content": submission,
                             "extra": {"exit_status": "Submitted", "submission": submission}})
        return output

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        # Commands execute in Pier's Linux task container, not on this macOS
        # host.  The task contract proves Linux but does not prove a kernel or
        # CPU architecture, so leave those fields blank rather than inventing
        # host-derived values for the prompt template.
        return {
            "cwd": self.cwd,
            "env": self.env,
            "timeout": self.timeout,
            "system": "Linux",
            "release": "",
            "version": "",
            "machine": "",
            **kwargs,
        }

    def serialize(self) -> dict[str, Any]:
        return {"info": {"config": {"environment": {
            "cwd": self.cwd, "env": self.env, "timeout": self.timeout,
        }, "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}"}}}


class DeepSWEHostMiniSweAgent(BaseAgent):
    """Pier adapter for mini-swe-agent installed on the host.

    ``api_base`` and ``api_key`` are host-only model settings.  For llama-server
    pass a full OpenAI-compatible URL (normally ending in ``/v1``) and a dummy
    non-secret API key if the server does not authenticate requests.
    """

    SUPPORTS_ATIF = True

    def __init__(
        self,
        *args: Any,
        config_file: str = "mini",
        config_yaml: str | None = None,
        api_base: str | None = None,
        api_key: str = "local",
        model_class: str = "litellm",
        model_kwargs: dict[str, Any] | None = None,
        step_limit: int | None = None,
        cost_limit: float | None = None,
        command_timeout: int | None = None,
        deterministic_outputs_file: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._config_file = config_file
        self._config_yaml = config_yaml
        self._api_base = api_base
        self._api_key = api_key
        self._model_class = model_class
        self._model_kwargs = model_kwargs or {}
        self._step_limit = step_limit
        self._cost_limit = cost_limit
        self._command_timeout = command_timeout
        self._deterministic_outputs_file = deterministic_outputs_file

    @staticmethod
    def name() -> str:
        return "deep-swe-host-mini-swe-agent"

    def version(self) -> str | None:
        import minisweagent
        return minisweagent.__version__

    async def setup(self, environment: BaseEnvironment) -> None:
        # minisweagent is deliberately installed on the host, not the sandbox.
        return None

    async def _detect_cwd(self, environment: BaseEnvironment) -> str:
        # The task image conventionally exposes the repository at one of these
        # two paths.  Do not use ``cd`` as a probe: both directories may exist
        # in an image even when neither is a worktree, and falling back to ``/``
        # would make a later commit/collection silently target the wrong tree.
        result = await environment.exec(
            command=(
                "for d in /app /testbed; do "
                "if [ -d \"$d\" ]; then "
                "root=$(git -C \"$d\" rev-parse --show-toplevel 2>/dev/null) || continue; "
                "if [ \"$root\" = \"$d\" ]; then printf '%s\\n' \"$d\"; exit 0; fi; "
                "fi; "
                "done; exit 42"
            ),
            timeout_sec=60,
        )
        cwd = (result.stdout or "").strip()
        if result.return_code != 0 or cwd not in _SUPPORTED_GIT_ROOTS:
            raise RuntimeError(
                "task image has no supported git worktree at /app or /testbed "
                f"(rc={result.return_code}, stderr={(result.stderr or '')[-1000:]!r})"
            )
        return cwd

    async def _validate_git_root(self, environment: BaseEnvironment, cwd: str) -> str:
        """Ensure a commit or evidence capture is rooted in the task worktree."""
        if cwd not in _SUPPORTED_GIT_ROOTS:
            raise RuntimeError(f"refusing Git operation outside supported task roots: {cwd!r}")
        result = await environment.exec(
            command=f"git -C {shlex.quote(cwd)} rev-parse --show-toplevel",
            timeout_sec=60,
        )
        root = (result.stdout or "").strip()
        if result.return_code != 0 or root != cwd:
            raise RuntimeError(
                f"refusing Git operation: {cwd!r} is not the validated task root "
                f"(rc={result.return_code}, root={root!r}, stderr={(result.stderr or '')[-1000:]!r})"
            )
        return root

    def _load_config(self) -> dict[str, Any]:
        from minisweagent.config import get_config_path

        if self._config_yaml is not None:
            config = yaml.safe_load(self._config_yaml)
        else:
            config = yaml.safe_load(get_config_path(self._config_file).read_text())
        if not isinstance(config, dict):
            raise ValueError("mini-swe-agent config must be a YAML mapping")
        return config

    def _build_model(self, config: dict[str, Any]):
        # Avoid LiteLLM's import-time refresh of a public cost map. Local model
        # accounting is deliberately best-effort for an offline benchmark host.
        os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "true")
        if self._deterministic_outputs_file:
            from minisweagent.models.test_models import DeterministicModel
            return DeterministicModel(outputs=json.loads(Path(self._deterministic_outputs_file).read_text()))

        if not self.model_name:
            raise ValueError("Pier requires --model openai/<local-model-name> for this adapter")
        from minisweagent.models import get_model

        model_config = dict(config.get("model") or {})
        model_config["model_name"] = self.model_name
        model_config["model_class"] = self._model_class
        model_config["cost_tracking"] = "ignore_errors"
        kwargs = dict(model_config.get("model_kwargs") or {})
        kwargs.update(self._model_kwargs)
        if self._api_base:
            kwargs["api_base"] = self._api_base
        # This value stays host-side.  It is only a local-server placeholder by default.
        if self._api_key:
            kwargs["api_key"] = self._api_key
        model_config["model_kwargs"] = kwargs
        return get_model(config=model_config)

    async def _commit_agent_work(self, environment: BaseEnvironment, cwd: str) -> None:
        # DeepSWE's verifier extracts ``git diff <base> HEAD``.  A working-tree
        # diff or staged change alone is intentionally not enough.
        await self._validate_git_root(environment, cwd)
        quoted_cwd = shlex.quote(cwd)
        command = (
            f"cd {quoted_cwd} && "
            "test \"$(git rev-parse --show-toplevel)\" = \"$PWD\" && "
            "git config --global --add safe.directory \"$PWD\" && "
            "if test -n \"$(git status --porcelain)\"; then "
            "git add -A && "
            "git -c user.name='DeepSWE host agent' -c user.email='deep-swe-host@local' "
            "commit -m 'deep-swe-agent submission'; "
            "fi"
        )
        result = await environment.exec(command=command, timeout_sec=120)
        if result.return_code != 0:
            raise RuntimeError("failed to commit agent work for DeepSWE collection: " + (result.stderr or result.stdout or "unknown error"))

    async def _capture_git_evidence(
        self, environment: BaseEnvironment, cwd: str, *, base_head: str | None = None,
    ) -> dict[str, Any]:
        """Record read-only Git evidence without changing Pier's collector contract.

        The verifier remains authoritative.  ``base_head`` is the task HEAD
        observed before agent execution, not a claim about an internal Pier
        verifier base that the adapter cannot inspect.
        """
        await self._validate_git_root(environment, cwd)
        quoted_cwd = shlex.quote(cwd)

        async def observe(label: str, command: str) -> tuple[str | None, dict[str, Any]]:
            try:
                result = await environment.exec(
                    command=f"cd {quoted_cwd} && git -c safe.directory={quoted_cwd} {command.removeprefix('git ')}",
                    timeout_sec=120,
                )
                value = (result.stdout or "").strip()
                return value, {"return_code": result.return_code, "stderr": (result.stderr or "").strip()}
            except Exception as exc:
                return None, {"return_code": None, "exception_type": type(exc).__name__, "exception": str(exc)}

        head, head_info = await observe("head", "git rev-parse --verify HEAD")
        status, status_info = await observe("status", "git status --porcelain")
        evidence: dict[str, Any] = {
            "cwd": cwd,
            "base_head": base_head,
            "head": head if head_info.get("return_code") == 0 else None,
            "head_observation": head_info,
            "status_porcelain": status if status_info.get("return_code") == 0 else None,
            "status_observation": status_info,
        }
        if base_head is not None and evidence["head"] is not None:
            # Git object IDs are generated by Git, but validate before placing
            # them into a command so a malformed observation is never executed.
            if all(re.fullmatch(r"[0-9a-fA-F]{40,64}", value or "") for value in (base_head, evidence["head"])):
                diff, diff_info = await observe("diff", f"git diff --binary {base_head} {evidence['head']}")
                evidence["diff_observation"] = diff_info
                if diff_info.get("return_code") == 0:
                    patch_path = self.logs_dir / "deep-swe-submission.patch"
                    patch_path.write_text(diff or "")
                    evidence["patch_file"] = patch_path.name
                    evidence["patch_bytes"] = len((diff or "").encode())
            else:
                evidence["diff_observation"] = {"return_code": None, "reason": "invalid git object id"}
        return evidence

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        from minisweagent.agents.default import DefaultAgent
        from pier.agents.installed.mini_swe_agent import convert_and_save_trajectory
        from pier.utils.trajectory_metrics import populate_context_from_final_metrics

        config = self._load_config()
        agent_config = dict(config.get("agent") or {})
        if self._step_limit is not None:
            agent_config["step_limit"] = int(self._step_limit)
        if self._cost_limit is not None:
            agent_config["cost_limit"] = float(self._cost_limit)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        native_path = self.logs_dir / "mini-swe-agent.trajectory.json"
        agent_config["output_path"] = native_path

        cwd = await self._detect_cwd(environment)
        git_before = await self._capture_git_evidence(environment, cwd)
        env_config = config.get("environment") or {}
        cancelled = threading.Event()
        sandbox = PierSandboxEnvironment(
            environment, asyncio.get_running_loop(), cwd=cwd,
            env=dict(env_config.get("env") or {}),
            # DeepSWE commands need the documented 120s budget by default;
            # do not silently inherit mini-swe-agent's historical 60s value.
            timeout=int(self._command_timeout if self._command_timeout is not None else 120),
            cancelled=cancelled,
        )
        model = CancellationAwareModel(self._build_model(config), cancelled)
        agent = DefaultAgent(model, sandbox, **agent_config)
        self.logger.info("deep-swe host agent: cwd=%s model=%s config=%s", cwd, self.model_name, self._config_file)
        worker = asyncio.create_task(asyncio.to_thread(agent.run, instruction))
        try:
            result = await asyncio.shield(worker)
        except asyncio.CancelledError:
            # Let an in-flight HTTP request finish in its thread, but make its
            # result unusable.  In particular, do not commit or create a
            # DeepSWE submission after the cancelled Pier trial returns.
            cancelled.set()

            def consume_worker_result(done: asyncio.Task[Any]) -> None:
                try:
                    done.result()
                except (WorkerCancelled, asyncio.CancelledError):
                    pass
                except Exception:
                    self.logger.exception("deep-swe mini-swe worker failed after cancellation")

            worker.add_done_callback(consume_worker_result)
            raise

        # The commit must happen before Pier invokes verifier.collect.
        await self._commit_agent_work(environment, cwd)
        git_after = await self._capture_git_evidence(environment, cwd, base_head=git_before.get("head"))
        (self.logs_dir / "deep-swe-git-evidence.json").write_text(json.dumps({
            "before": git_before, "after": git_after,
            "note": "base_head is the task HEAD observed before the agent; Pier verifier remains authoritative",
        }, indent=2))
        (self.logs_dir / "result.json").write_text(json.dumps({
            "exit_status": result.get("exit_status"), "n_calls": agent.n_calls,
            "cost": agent.cost, "model": self.model_name, "api_base": self._api_base,
        }, indent=2))
        (self.logs_dir / "host-agent.json").write_text(json.dumps({
            "adapter": self.name(), "mini_swe_agent": self.version(), "model": self.model_name,
            "native_trajectory": native_path.name, "commit_required": True,
        }, indent=2))

        if native_path.exists():
            atif_path = self.logs_dir / "trajectory.json"
            try:
                trajectory = convert_and_save_trajectory(native_path, atif_path, str(uuid.uuid4()))
                if trajectory.final_metrics:
                    populate_context_from_final_metrics(context, trajectory.final_metrics)
            except Exception as exc:
                # Verifier collection uses the committed repository state, not
                # ATIF. Keep conversion failure observable without invalidating
                # an otherwise scored trial.
                (self.logs_dir / "atif-conversion-error.json").write_text(json.dumps({
                    "exception_type": type(exc).__name__, "exception": str(exc),
                    "native_trajectory": native_path.name, "atif_trajectory": atif_path.name,
                }, indent=2))
                self.logger.exception("ATIF conversion failed after agent work was committed")
        context.metadata = {**(context.metadata or {}), "host_side_model": True, "working_directory": cwd}
