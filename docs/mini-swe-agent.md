# SchwartzStar coding agent with mini-SWE-agent 2.4.6

This is a supervised, local coding workflow for an already-running
OpenAI-compatible SchwartzStar endpoint. It uses mini-SWE-agent's LiteLLM
model adapter and its local environment. The guide does not start a model
server, run an inference, build a container, or grant unattended write
access.

## What this setup assumes

- mini-SWE-agent **2.4.6** is installed in a Python environment.
- SchwartzStar is already listening on loopback and exposes an OpenAI-style
  `/v1/chat/completions` endpoint with tool calls enabled.
- The endpoint's model is identified as `inkling-2-mini`.
- The repository is a deliberate working copy. The agent may run shell
  commands and edit files in its configured working directory.

The `openai/` prefix in `openai/inkling-2-mini` selects LiteLLM's OpenAI
provider. `api_base` is the **server root**, for example
`http://127.0.0.1:8080/v1`; do not append `/chat/completions` to it.

Start the server with the [quick start](../README.md#quick-start), then use a
second terminal for the agent.

## A first supervised run

Set the virtual-environment path and repository path to local values. The key
below is deliberately a non-secret placeholder; replace it only with a real
secret if the loopback server requires authentication, preferably in the
environment rather than in a config file.

```bash
MSWEA_VENV=/absolute/path/to/virtualenv
REPO=/absolute/path/to/a/working-copy

python3 -m venv "$MSWEA_VENV"
"$MSWEA_VENV/bin/python" -m pip install 'mini-swe-agent==2.4.6'
```

Then start the interactive agent:

```bash
cd "$REPO"
OPENAI_API_KEY=local-placeholder \
MSWEA_CONFIGURED=1 \
MSWEA_COST_TRACKING=ignore_errors \
"$MSWEA_VENV/bin/mini-swe-agent" \
  -c mini.yaml \
  -c model.model_class=litellm \
  -c model.model_name=openai/inkling-2-mini \
  -c model.model_kwargs.api_base=http://127.0.0.1:8080/v1 \
  -c model.model_kwargs.max_tokens=2048 \
  -c model.cost_tracking=ignore_errors \
  -c environment.cwd="$REPO" \
  -c environment.timeout=60 \
  -c agent.mode=confirm \
  -c agent.step_limit=24 \
  -c agent.cost_limit=0 \
  -c agent.wall_time_limit_seconds=600 \
  -o /tmp/schwartzstar-mini-run.traj.json \
  --task 'Inspect the focused issue, make the smallest compatible fix, and run only the relevant tests. Do not touch unrelated files or use destructive commands.'
```

`mini.yaml` is the v2 interactive configuration shipped with mini-SWE-agent;
passing it explicitly matters because supplying `-c` specs replaces the
implicit default config. The command keeps `confirm` mode, so commands not on
its whitelist are shown for review. Use a task-specific repository copy and
inspect the proposed commands before running them.

The trajectory is JSON and can contain prompts, model responses, paths, and
commands. Treat it as sensitive and inspect or redact it before sharing.

## Configuring a local endpoint

The same settings can be kept in a private YAML file instead of repeated CLI
overrides. Do not commit credentials. A minimal shape is:

```yaml
model:
  model_class: litellm
  model_name: openai/inkling-2-mini
  model_kwargs:
    api_base: http://127.0.0.1:8080/v1
  # A custom local model is often absent from LiteLLM's cost registry.
  cost_tracking: ignore_errors
environment:
  cwd: /absolute/path/to/a/working-copy
  timeout: 60
agent:
  mode: confirm
  step_limit: 24
  cost_limit: 0
  wall_time_limit_seconds: 600
```

When the model has a LiteLLM cost entry, use `cost_tracking: default` and, if
needed, set `litellm_model_registry` to a private registry JSON file. With an
unregistered local model, strict default tracking can fail after a successful
response because mini-SWE-agent asks LiteLLM to calculate a price. The
`ignore_errors` setting prevents that failure, but it reports no reliable
spend and makes a cost limit ineffective; keep step and wall-time limits.

`OPENAI_API_KEY` is read by the OpenAI/LiteLLM provider. For an unauthenticated
loopback server, a placeholder may be sufficient; for an authenticated one,
export the real value in the invoking shell or use a secret manager. Never
paste a real key into this guide, a committed YAML file, or a trajectory.

## Host execution and isolation boundaries

The installed `local` environment executes each action directly on the host
with a subprocess and the configured `cwd`; it is not an OS sandbox. The
interactive agent's `confirm` mode is a human approval gate, not a security
boundary. Keep it enabled for normal work and reject commands that are
destructive, broad, or outside the task.

Some internal smoke tests used a custom Seatbelt environment. That is a
separate harness choice and is not created automatically by this CLI. Do not
confuse `local` with that Seatbelt wrapper, and do not add an unverified
`--environment-class` value. Docker is not required by this workflow.

## Context and reasoning limits

An 8K context is a smoke/debug bound, not a repository-scale working context.
Use a focused checkout, a narrow task, and explicit files/tests; do not ask the
agent to ingest an entire repository. Split large repairs and compact the
problem statement between runs. A long command output is also context, so
prefer targeted `rg`, `sed`, and test commands.

The `coding` server profile defaults to a 128-token reasoning budget. Small
smoke tests support this starting point but do not establish an optimum for
complex repairs. Set `INKLING_REASONING_BUDGET` when starting the server to
compare budgets, and leave room for reasoning and final output in the client's
generation limit.

This workflow is a supervised smoke path. Parser success, a responding
endpoint, or one successful edit does not establish coding quality, patch
correctness, performance, or SWE-bench/mini-SWE-agent suite results. No suite
or benchmark is promised by this guide. These CLI settings have been checked
with the 2.4.6 configuration parser; the exact interactive recipe has not yet
been validated end to end. The reported agent smokes used a separate sandboxed
environment.

## Useful checks before a run

These checks do not start a server or call a model:

```bash
"$MSWEA_VENV/bin/mini-swe-agent" --help
"$MSWEA_VENV/bin/mini-swe-agent" --help | rg 'model|config|environment|output|cost'
```

The official mini-SWE-agent
documentation covers the CLI and local-model details:

- [mini-SWE-agent CLI](https://mini-swe-agent.com/latest/usage/mini/)
- [local models and cost tracking](https://mini-swe-agent.com/latest/models/local_models/)
- [mini-SWE-agent repository](https://github.com/SWE-agent/mini-swe-agent)
- [LiteLLM OpenAI provider](https://docs.litellm.ai/docs/providers/openai)

