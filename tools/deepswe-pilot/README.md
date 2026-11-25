# Portable DeepSWE one-task runner

This is an isolated runner for one DeepSWE task through Pier. It is designed
for an Apple Silicon host with a separately managed local OpenAI-compatible
server. It does not start that server, Docker, or a model during preflight.

The runner preserves the task's own agent timeout, agent no-network policy,
and separate no-network verifier. It does not provide a verification bypass or
replace the official Pier collector. The included `deep_host_mini_swe.py` is
the cancellation-safe host adapter used by default. Use `--adapter PATH` only
to deliberately override it; the selected adapter SHA-256 is always recorded.

`mini-local.example.yaml` carries the complete mini-SWE-agent 2.4.6 prompt
used by this host harness. The adapter supplies `api_base` and its local dummy
`api_key`, so neither appears in the template. This aligns prompt structure
within this harness only; it does not establish equivalence with any published
benchmark configuration or result.

## Example

```sh
PYTHON=/path/to/venv/bin/python
TASKS=/path/to/deep-swe/tasks
OUT=/path/to/new-empty-output

"$PYTHON" run_one_task.py \
  --python "$PYTHON" --tasks-dir "$TASKS" \
  --task-id superjson-error-stack-serialization \
  --config mini-local.example.yaml \
  --model openai/local-coding-model --server-url http://127.0.0.1:8000/v1 \
  --output-dir "$OUT" --dry-run
```

Remove `--dry-run` only when Pier, Docker images, task data, and the local
server are deliberately ready. The default job is single-concurrency and
enables verification. The adapter runs model requests on the host while shell
actions remain inside the task's no-network container.

## Reproducible local setup

The task repository used for this harness is
[`datacurve-ai/deep-swe`](https://github.com/datacurve-ai/deep-swe.git) at
commit `0b9fabbb63b9104d678fe965e1632f2dd9eaa2ea`.

```sh
git clone https://github.com/datacurve-ai/deep-swe.git deep-swe
cd deep-swe
git checkout 0b9fabbb63b9104d678fe965e1632f2dd9eaa2ea

python3.14 -m venv .venv
.venv/bin/pip install \
  'datacurve-pier==0.3.1' 'mini-swe-agent==2.4.6' \
  'litellm==1.110.3' 'PyYAML==6.0.4'
```

Those are the dependency versions tested with this candidate. Use Python 3.11
or newer, retain the installation evidence for a real run, and confirm the
selected runtime can load Pier with `"$PYTHON" invoke_pier.py --help` before
attempting a task.

For an end-to-end local test, the user must first start Docker Desktop and a
non-secret loopback OpenAI-compatible server. The tested endpoint/model shape
is `http://127.0.0.1:8080/v1` and `openai/inkling-2-mini`. Confirm server
health before preflight, then run this candidate from its own directory:

```sh
curl --fail http://127.0.0.1:8080/health

PYTHON=/path/to/.venv/bin/python
TASKS=/path/to/deep-swe/tasks
OUT=/path/to/new-empty-output

"$PYTHON" run_one_task.py \
  --python "$PYTHON" --tasks-dir "$TASKS" \
  --task-id superjson-error-stack-serialization \
  --config mini-local.example.yaml \
  --model openai/inkling-2-mini --server-url http://127.0.0.1:8080/v1 \
  --output-dir "$OUT" --dry-run
```

Inspect `run-manifest.json` and `result-summary.json`, then remove `--dry-run`
only when deliberately ready to start one Docker-backed Pier task. The runner
does not start Docker or the model server and its dry-run does not itself call
the endpoint.

The runner deliberately has no credential flag or environment forwarding. It
uses the adapter's dummy local API key and accepts only a loopback endpoint
without user info, query parameters, or fragments. The model identifier and
endpoint must be non-secret: Pier receives them as command arguments and may
write them into its own process metadata, logs, or job configuration. The
runner redacts them only in its own manifest; do not put credentials into a
model name, endpoint URL, or YAML config.

## Evidence

The new or empty output directory receives:

- `run-manifest.json`: task id/base commit, SHA-256 values for task, adapter,
  and config, selected-interpreter dependency versions, and hashes (not values)
  of the model identifier and endpoint. This manifest does not make Pier's
  own logs or job configuration secret-safe.
- `pier.log`: Pier output.
- `result-summary.json`: exception presence and exception value type (never the
  raw exception text), task-identity verification, the reward reported in the
  trial result, the reward in `reward.json`, and a fail-closed status.

Dry-run evidence is preflight evidence only: it verifies inputs, task policy,
adapter/config hashes, and selected-runtime dependencies. It is not a Pier
trial and does not create a task result, patch, verifier reward, Docker
container, model request, or server process.

The repository's mock tests exercise the bundled adapter's task-root,
commit-collection, cancellation, and post-run evidence contracts against the
real selected Python environment. They use fake environments and deterministic
model stubs; they do not invoke Pier jobs, Docker, model endpoints, or servers.

This can record one Pier task execution only. It does not establish benchmark
reproducibility, hardware parity, full-suite validity, or any published score.
Compare task images, task TOMLs, adapter/config hashes, runtime versions,
server build/model, and raw verifier artifacts before making any comparison.

The deterministic fixture end-to-end check completed two tool calls, produced
a 195-byte patch, and recorded verifier reward `0`. That is a harness protocol
check, not a solved task or benchmark result. No real benchmark campaign has
been completed with this candidate; do not compare this check or its reward to
published DeepSWE, SWE-bench, or other benchmark scores.

## Constraints

- The selected Python must import `pier`, `minisweagent`, `litellm`, and
  `yaml`; Python 3.11 or newer is required. The runner resolves the `pier`
  console-script entry point from the selected Python's `datacurve-pier`
  distribution (Pier does not provide `python -m pier`) and records its Python
  hash and detected Pier distribution version.
- The runner preserves the task's declared budget rather than imposing a
  portable replacement timeout.
- The runner's manifest stores model and endpoint hashes only, but Pier may
  store the raw non-secret command values in its log or job configuration.
- Literal YAML credential-like keys are rejected, except the exact bundled
  dummy placeholder `api_key: local`, which the adapter overwrites with its
  own local placeholder. This scanner is heuristic, not a secret audit of an
  adapter override, Pier logs, task files, or job artifacts.

## Deterministic local OpenAI fixture

`tests/openai_fixture.py` is an optional loopback-only fixture for a future
end-to-end adapter check. It returns exactly two `bash` tool calls: one creates
`mini_swe_fixture.txt` in the task container; the second submits the task.
It has no credentials, model weights, Docker dependency, or cost calculation.

Start it in a separate terminal only when a test window is available:

```sh
python3 tests/openai_fixture.py --port 18080 --log tests/openai-fixture.log
```

It binds only `127.0.0.1`. Confirm it with `curl http://127.0.0.1:18080/health`.
Stop it with `Ctrl-C`; the JSONL log records only lifecycle, health, and request
sequence events, never request bodies or headers.
