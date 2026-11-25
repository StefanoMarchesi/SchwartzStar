"""Local tests for parsing, redaction, and result classification only."""
from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("runner", ROOT / "run_one_task.py")
runner = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(runner)


def main() -> None:
    redacted = runner.redact_endpoint("http://127.0.0.1:9999/v1")
    assert redacted["scheme"] == "http" and "127.0.0.1" not in json.dumps(redacted)
    for unsafe in ("http://user:secret@127.0.0.1/v1", "http://127.0.0.1/v1?token=secret", "http://example.test/v1"):
        try:
            runner.redact_endpoint(unsafe)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe endpoint accepted: {unsafe}")
    for unsafe in ("openai/secret@model", "https://secret.example/model", "openai/model?token=secret"):
        try:
            runner.validate_model_identifier(unsafe)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe model accepted: {unsafe}")
    assert runner.validate_model_identifier("openai/local-coding-model") == "openai/local-coding-model"
    assert runner.PIER_INVOKER.name == "invoke_pier.py" and runner.PIER_INVOKER.is_file()
    assert not runner.numeric(float("nan")) and not runner.numeric(float("inf"))
    for unsafe in ("../escape", "name/escape", ".hidden", "name..again"):
        try:
            runner.validate_identifier(unsafe, "--job-name")
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe job name accepted: {unsafe}")
    with tempfile.TemporaryDirectory() as temporary:
        job = Path(temporary) / "job" / "trial"
        (job / "verifier").mkdir(parents=True)
        good = {"exception_info": None, "task_name": "example/task", "verifier": {"finished_at": "x"}, "verifier_result": {"rewards": {"reward": 1}}}
        (job / "result.json").write_text(json.dumps(good))
        (job / "verifier" / "reward.json").write_text(json.dumps({"reward": 1}))
        assert runner.collect_result(job.parent, 0, "example/task")["status"] == "completed_with_consistent_verifier_reward"
        (job / "result.json").write_text(json.dumps({**good, "exception_info": "timeout"}))
        assert runner.collect_result(job.parent, 0, "example/task")["status"] == "trial_exception"
        assert runner.collect_result(job.parent, 0, "example/task")["trial_exception_kind"] == "str"
        (job / "result.json").write_text(json.dumps({**good, "task_name": "wrong/task"}))
        assert runner.collect_result(job.parent, 0, "example/task")["status"] == "task_identity_mismatch"
        (job / "result.json").write_text(json.dumps({key: value for key, value in good.items() if key != "verifier"}))
        assert runner.collect_result(job.parent, 0, "example/task")["status"] == "verifier_missing"
        (job / "result.json").write_text(json.dumps({**good, "verifier_result": {"rewards": {}}}))
        assert runner.collect_result(job.parent, 0, "example/task")["status"] == "verifier_reward_invalid"
        (job / "result.json").write_text(json.dumps({**good, "verifier_result": {"rewards": {"reward": True}}}))
        assert runner.collect_result(job.parent, 0, "example/task")["status"] == "verifier_reward_invalid"
        (job / "result.json").write_text(json.dumps(good))
        (job / "verifier" / "reward.json").unlink()
        assert runner.collect_result(job.parent, 0, "example/task")["status"] == "reward_file_missing"
        (job / "verifier" / "reward.json").write_text("not json")
        assert runner.collect_result(job.parent, 0, "example/task")["status"] == "reward_file_invalid"
        (job / "verifier" / "reward.json").write_text(json.dumps({"reward": "one"}))
        assert runner.collect_result(job.parent, 0, "example/task")["status"] == "reward_file_invalid"
        (job / "verifier" / "reward.json").write_text(json.dumps([1]))
        assert runner.collect_result(job.parent, 0, "example/task")["status"] == "reward_file_invalid"
        (job / "verifier" / "reward.json").write_text(json.dumps({"reward": 0}))
        assert runner.collect_result(job.parent, 0, "example/task")["status"] == "reward_mismatch"
        (job / "verifier" / "reward.json").write_text(json.dumps({"reward": 1}))
        assert runner.collect_result(job.parent, 3, "example/task")["status"] == "pier_failed"
        (job / "result.json").write_text("not json")
        assert runner.collect_result(job.parent, 0, "example/task")["status"] == "malformed_trial_result"
    with tempfile.TemporaryDirectory() as temporary:
        config = Path(temporary) / "config.yaml"
        config.write_text("model:\n  timeout: 30\n")
        runner.reject_explicit_config_credentials(config)
        config.write_text("model:\n  api_key: local\n")
        runner.reject_explicit_config_credentials(config)
        config.write_text("model:\n  api_key: secret\n")
        try:
            runner.reject_explicit_config_credentials(config)
        except ValueError:
            pass
        else:
            raise AssertionError("credential-like config key was accepted")
    with tempfile.TemporaryDirectory() as temporary:
        tasks = Path(temporary) / "tasks" / "one"
        tasks.mkdir(parents=True)
        tasks.joinpath("task.toml").write_text('''[metadata]\ntask_id = "one"\nbase_commit_hash = "a"\n[agent]\ntimeout_sec = 1\nnetwork_mode = "no-network"\n[verifier]\ntimeout_sec = 1\nnetwork_mode = "no-network"\nenvironment_mode = "separate"\n''')
        path, task = runner.read_task(tasks.parent, "one")
        assert path.name == "task.toml" and task["metadata"]["task_id"] == "one"
        tasks.joinpath("task.toml").write_text(tasks.joinpath("task.toml").read_text().replace('environment_mode = "separate"', 'environment_mode = "same"'))
        try:
            runner.read_task(tasks.parent, "one")
        except ValueError as exc:
            assert "separate no-network verifier" in str(exc)
        else:
            raise AssertionError("same-environment verifier was accepted")
    print("runner mocks: ok")


if __name__ == "__main__":
    main()
