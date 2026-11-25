#!/usr/bin/env python3
"""Run exactly one DeepSWE/Pier task and record portable, non-secret evidence.

This launcher intentionally does not start a model server or alter a task.  It
delegates task execution and the official separate verifier to Pier.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, packages_distributions, version
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

DEFAULT_ADAPTER = Path(__file__).with_name("deep_host_mini_swe.py")
PIER_INVOKER = Path(__file__).with_name("invoke_pier.py")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_report(python: Path) -> dict[str, Any]:
    """Ask the selected interpreter, not this launcher, about its runtime."""
    probe = r'''import importlib.util, json, sys
from importlib.metadata import PackageNotFoundError, packages_distributions, version
names = {"Pier": ("pier", "datacurve-pier"), "mini-swe-agent": ("minisweagent", "mini-swe-agent"), "LiteLLM": ("litellm", "litellm"), "PyYAML": ("yaml", "PyYAML")}
out = {"python": sys.version.split()[0], "executable": sys.executable, "packages": {}}
mapping = packages_distributions()
for label, (module, fallback) in names.items():
    distributions = mapping.get(module, [fallback])
    found = []
    for distribution in distributions:
        try: found.append({"distribution": distribution, "version": version(distribution)})
        except PackageNotFoundError: pass
    out["packages"][label] = {"module_importable": importlib.util.find_spec(module) is not None, "distributions": found}
print(json.dumps(out, sort_keys=True))'''
    completed = subprocess.run([str(python), "-c", probe], text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(f"dependency probe failed: {completed.stderr.strip()}")
    return json.loads(completed.stdout)


def redact_endpoint(value: str) -> dict[str, str]:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("--server-url must be an absolute http(s) URL")
    if parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
        raise ValueError("--server-url must not contain credentials, query parameters, or fragments")
    if parsed.hostname not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("--server-url must target a loopback host")
    return {"sha256": hashlib.sha256(value.encode()).hexdigest(), "scheme": parsed.scheme}


def validate_identifier(value: str, label: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value) or ".." in value:
        raise ValueError(f"{label} must be a simple identifier, without traversal")
    return value


def validate_model_identifier(value: str) -> str:
    if not value or value != value.strip() or any(marker in value for marker in ("://", "@", "?", "#")):
        raise ValueError("--model must be a non-secret model identifier, not a URL or credential")
    return value


def reject_explicit_config_credentials(path: Path) -> None:
    """Reject obvious literal credential keys; this is not a full secret audit."""
    for match in re.finditer(r"(?im)^\s*(api[_-]?key|token|password|secret)\s*:\s*(.*?)\s*$", path.read_text()):
        key, value = match.group(1).lower().replace("-", "_"), match.group(2)
        # The bundled adapter defaults to api_key="local" and overwrites the
        # config's model_kwargs api_key with that same host-only placeholder.
        if key == "api_key" and value == "local":
            continue
        raise ValueError("--config contains an explicit credential-like YAML key")


def numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def read_task(tasks_dir: Path, task_id: str) -> tuple[Path, dict[str, Any]]:
    path = tasks_dir / task_id / "task.toml"
    if not path.is_file():
        raise FileNotFoundError(f"task.toml is missing for task {task_id!r}: {path}")
    task = tomllib.loads(path.read_text())
    metadata = task.get("metadata") or {}
    verifier = task.get("verifier") or {}
    agent = task.get("agent") or {}
    if metadata.get("task_id") != task_id:
        raise ValueError("task_id must match task.toml metadata.task_id")
    if verifier.get("environment_mode") != "separate" or verifier.get("network_mode") != "no-network":
        raise ValueError("refusing task without separate no-network verifier")
    if agent.get("network_mode") != "no-network":
        raise ValueError("refusing task without no-network agent policy")
    if not isinstance(agent.get("timeout_sec"), (int, float)):
        raise ValueError("task.toml agent.timeout_sec must be numeric")
    return path, task


def collect_result(job_dir: Path, pier_exit: int, expected_task_name: str) -> dict[str, Any]:
    trials = sorted(item for item in job_dir.iterdir() if item.is_dir() and (item / "result.json").is_file()) if job_dir.is_dir() else []
    summary: dict[str, Any] = {"pier_exit": pier_exit, "trial_count": len(trials), "trial_exception_present": False, "trial_exception_kind": None, "task_identity_verified": False, "verifier_present": False, "reported_verifier_reward": None, "reward_file_value": None}
    if len(trials) != 1:
        summary["status"] = "invalid_trial_count"
        return summary
    try:
        result = json.loads((trials[0] / "result.json").read_text())
    except (OSError, json.JSONDecodeError):
        summary["status"] = "malformed_trial_result"
        return summary
    if not isinstance(result, dict):
        summary["status"] = "malformed_trial_result"
        return summary
    exception = result.get("exception_info")
    summary["trial_exception_present"] = exception is not None
    summary["trial_exception_kind"] = None if exception is None else type(exception).__name__
    summary["task_identity_verified"] = result.get("task_name") == expected_task_name
    verifier = result.get("verifier")
    summary["verifier_present"] = verifier is not None
    verifier_result = result.get("verifier_result")
    rewards = verifier_result.get("rewards") if isinstance(verifier_result, dict) else None
    if isinstance(rewards, dict) and numeric(rewards.get("reward")):
        summary["reported_verifier_reward"] = rewards["reward"]
    reward_file = trials[0] / "verifier" / "reward.json"
    reward_data: Any = None
    if reward_file.is_file():
        try:
            reward_data = json.loads(reward_file.read_text())
        except (OSError, json.JSONDecodeError):
            reward_data = None
            reward_file_malformed = True
        else:
            reward_file_malformed = not isinstance(reward_data, dict)
        if isinstance(reward_data, dict) and numeric(reward_data.get("reward")):
            summary["reward_file_value"] = reward_data["reward"]
    else:
        reward_file_malformed = False
    if pier_exit != 0:
        summary["status"] = "pier_failed"
    elif summary["trial_exception_present"]:
        summary["status"] = "trial_exception"
    elif not summary["task_identity_verified"]:
        summary["status"] = "task_identity_mismatch"
    elif not summary["verifier_present"]:
        summary["status"] = "verifier_missing"
    elif not isinstance(verifier_result, dict) or not isinstance(rewards, dict) or not numeric(rewards.get("reward")):
        summary["status"] = "verifier_reward_invalid"
    elif not reward_file.is_file():
        summary["status"] = "reward_file_missing"
    elif reward_file_malformed or not isinstance(reward_data, dict) or not numeric(reward_data.get("reward")):
        summary["status"] = "reward_file_invalid"
    elif summary["reward_file_value"] != summary["reported_verifier_reward"]:
        summary["status"] = "reward_mismatch"
    else:
        summary["status"] = "completed_with_consistent_verifier_reward"
    return summary


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True, help="Python interpreter that owns Pier and the adapter dependencies")
    parser.add_argument("--tasks-dir", type=Path, required=True, help="Directory containing task-id/task.toml")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER, help="Host-side cancellation-safe adapter (defaults to bundled adapter)")
    parser.add_argument("--config", type=Path, required=True, help="mini-swe-agent YAML config")
    parser.add_argument("--model", required=True, help="Non-secret OpenAI-style model identifier")
    parser.add_argument("--server-url", required=True, help="Non-secret loopback OpenAI-compatible endpoint")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--job-name", default="portable-one-task")
    parser.add_argument("--dry-run", action="store_true", help="Write preflight evidence without invoking Pier")
    args = parser.parse_args()

    try:
        args.task_id = validate_identifier(args.task_id, "--task-id")
        args.job_name = validate_identifier(args.job_name, "--job-name")
        args.model = validate_model_identifier(args.model)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    for label, path in {"python": args.python, "adapter": args.adapter, "config": args.config, "tasks directory": args.tasks_dir}.items():
        if not path.exists():
            raise SystemExit(f"missing {label}: {path}")
    try:
        reject_explicit_config_credentials(args.config)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"output directory must be new or empty: {args.output_dir}")
    if not args.adapter.stem.isidentifier():
        raise SystemExit("adapter filename stem must be a Python module identifier")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    task_path, task = read_task(args.tasks_dir, args.task_id)
    dependencies = package_report(args.python)
    runtime_version = tuple(int(part) for part in dependencies["python"].split(".")[:2])
    if runtime_version < (3, 11):
        raise SystemExit("selected --python must be version 3.11 or newer")
    required = ("Pier", "mini-swe-agent", "LiteLLM", "PyYAML")
    unavailable = [name for name in required if not dependencies["packages"][name]["module_importable"]]
    endpoint = redact_endpoint(args.server_url)
    task_name = (task.get("task") or {}).get("name")
    if not isinstance(task_name, str) or not task_name:
        raise SystemExit("task.toml task.name must be a non-empty string")
    record = {
        "format": "portable-deepswe-one-task-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": {"id": args.task_id, "base_commit": (task.get("metadata") or {}).get("base_commit_hash"), "task_toml_sha256": sha256(task_path)},
        "adapter": {"sha256": sha256(args.adapter), "filename": args.adapter.name},
        "config": {"sha256": sha256(args.config), "filename": args.config.name},
        "model": {"sha256": hashlib.sha256(args.model.encode()).hexdigest()},
        "endpoint": endpoint,
        "task_policy": {"agent_timeout_sec": (task.get("agent") or {}).get("timeout_sec"), "agent_network_mode": (task.get("agent") or {}).get("network_mode"), "verifier_timeout_sec": (task.get("verifier") or {}).get("timeout_sec"), "verifier_environment_mode": (task.get("verifier") or {}).get("environment_mode"), "verifier_network_mode": (task.get("verifier") or {}).get("network_mode")},
        "dependency_check": dependencies,
        "pier_runtime": {"invocation": "selected-python invoke_pier.py", "python_sha256": sha256(args.python), "distribution": dependencies["packages"]["Pier"]},
        "secret_handling": "literal credential-like YAML keys are rejected; adapter, Pier logs, and job files are not asserted secret-free",
        "exceptions_vs_reward": "result-summary.json records exception presence/type separately from verifier reward values",
    }
    write_json(args.output_dir / "run-manifest.json", record)
    if unavailable:
        write_json(args.output_dir / "result-summary.json", {"status": "dependency_check_failed", "missing_modules": unavailable, "trial_exception_present": False, "trial_exception_kind": None, "reported_verifier_reward": None, "reward_file_value": None})
        raise SystemExit("missing required importable dependencies: " + ", ".join(unavailable))
    if args.dry_run:
        write_json(args.output_dir / "result-summary.json", {"status": "dry_run", "trial_exception_present": False, "trial_exception_kind": None, "reported_verifier_reward": None, "reward_file_value": None})
        return 0

    jobs_dir = args.output_dir / "jobs"
    import_path = f"{args.adapter.stem}:DeepSWEHostMiniSweAgent"
    command = [str(args.python), str(PIER_INVOKER), "run", "-p", str(args.tasks_dir), "--include-task-name", args.task_id, "--env", "docker", "--n-concurrent", "1", "--agent-import-path", import_path, "--model", args.model, "--agent-kwarg", f"config_file={args.config}", "--agent-kwarg", f"api_base={args.server_url}", "--no-force-build", "--delete", "--jobs-dir", str(jobs_dir), "--job-name", args.job_name, "--enable-verification"]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(args.adapter.parent) + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    environment.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "true")
    with (args.output_dir / "pier.log").open("w") as log:
        process = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
    summary = collect_result(jobs_dir / args.job_name, process.returncode, task_name)
    write_json(args.output_dir / "result-summary.json", summary)
    return 0 if summary["status"] == "completed_with_consistent_verifier_reward" else 1


if __name__ == "__main__":
    raise SystemExit(main())
