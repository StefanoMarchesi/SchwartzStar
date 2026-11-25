# Inkling 2 Mini — first coding-agent test

24 November 2026, MacBook Pro M1 Pro with 16 GB. IQ4_NL on Metal, microbatch 4, 1.536 MiB expert cache, four threads, zero reasoning budget. One server and a single inference queue. Context 4.096 for the initial run and 8.192 for the repeat. Process threshold **4.500 MiB**. The Git fixture is synthetic and contains one unique random marker, a Python `clamp` function, and a test that fails at the upper bound.

## Reproduction

Start the server with the profile in [runtime history](runtime-history.md). For the 8K repeat, change only `INKLING_CTX=8192`. From another terminal:

```bash
pid=$(lsof -tiTCP:18080 -sTCP:LISTEN)
probe_dir=$(mktemp -d "${TMPDIR:-/tmp}/inkling-agent.XXXXXX")
python3 -B tools/agent_probe.py "$probe_dir/codex" \
  --pid "$pid" --footprint "./build/bin/inkling-footprint" \
  --compact-codex
python3 -B tools/agent_probe.py "$probe_dir/opencode" \
  --pid "$pid" --footprint "./build/bin/inkling-footprint" \
  --client opencode --context 8192
```

The fixture creates a new directory for every run. Codex uses per-invocation overrides, a local provider with `wire_api="responses"`, read-only sandboxing for the first turn, and workspace-write for the second. The compact profile isolates the configuration and uses short instructions and shell tools. OpenCode uses Chat Completions with reading and repair agents limited to the fixture. The explicit title avoids the automatic title request. The local proxy records requests and SSE events without rewriting bodies or forwarding credentials. Raw logs remain outside the source tree.

Success criteria. Reading succeeds if the assistant repeats the marker exactly, the result is sent back with the call ID, and the fixture remains unchanged. Repair succeeds if only the function in `calc.py` changes, the agent log shows the test execution, and the independent verification also passes. An HTTP error, a missing token, or exceeding the footprint does not count as success.

## Codex CLI 0.171.0, ordinary configuration

| Context | Input expected by server | Input evaluated | Input reused | First text / tool | Turn time | Footprint |
| ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 4.096 | 17.658 | 0 | 0 | — | 2,03 s | 2.043,6 MiB |
| 8.192 | 17.658 | 0 | 0 | — | 1,34 s | 2.143,7 MiB |
| 8.192, isolated state | 13.874 | 0 | 0 | — | 0,27 s | 2.143,9 MiB |

All three attempts received HTTP 400 `exceed_context_size_error` before evaluation. The direct 8K test kept the request bytes unchanged. The initial payload contained the Codex instructions, skill catalog, and session tools. The isolated state removed connectors; instructions and catalog remained. For these rejections, tok/s and time to the first event do not apply. Swap was unchanged at 8,3 MiB in all three tests. macOS estimate of free memory before → after: 51 → 49%, 30 → 30%, and 36 → 35%, respectively.

## OpenCode 1.19.6, Chat Completions

The first invocation, without an explicit title, also queued 612 tokens for the conversation title. Title generation was interrupted after 203,6 s, before the single slot completed the reading task. The repeat uses a predefined title.

### Reading turn

| Request | Input | Re-evaluated | Reused | Prefill | First text | First tool call | API time | Decode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Read the file | 752 | 752 | 0 | 189,11 s | — | 195,31 s | 217,70 s | 2,45 tok/s |
| Interpret the result | 924 | 176 | 748 | 39,30 s | 49,52 s | — | 60,28 s | 3,01 tok/s |

From the start of the turn, the first tool event arrived at **197,75 s** and the first text at **269,86 s**. The full turn took **280,80 s**. Total prefill 228,41 s. Aggregate decode 2,73 tok/s over 135 tokens. Maximum process footprint at the end of the turn **2.637,3 MiB**. macOS estimate of free memory 34 → 28%, swap 213,1 → 214,1 MiB. The footprint measurement is the maximum recorded by macOS since server startup.

The SSE response contained a structured `read` call for `marker.txt`. The client log recorded it as completed. The following request contains a `role:"tool"` message with the file string and the same `tool_call_id`. The final text matches the fixture marker exactly. The reading-turn diff is empty. The KV checkpoint reused **748/924 tokens (81%)** in the second request.

## Guided two-turn PHP benchmark at 8K

Both models used the same synthetic PHP fixture, local llama.cpp build, Metal, one slot, four threads, 8,192 positions, microbatch 512, and two context checkpoints. Gemma 4 Q4_0 used resident weights. Inkling used SSD experts and a 3,072 MiB cache. Both used greedy sampling, at most two attempts per turn, and 512 output tokens per attempt. The controller restricts edits to the two source files, runs PHP without network or file writes, and checks the first test again after the credit change. Both models receive equivalent JSON edit instructions. Inkling starts the assistant's JSON content block with `{` to avoid repeatedly opening empty reasoning blocks.

| Model | Turn 1 to green | Turn 2 to green | Requests | Peak process footprint | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Gemma 4 12B Instruct Q4_0, resident | 23.55 s | 58.51 s | 1 + 2 | 1,777 MiB | Both turns and regression passed |
| Inkling 2 Mini UD-IQ2_XXS, SSD | 62.19 s | 95.71 s | 1 + 1 | 4,290 MiB | Both turns and regression passed |
| Inkling 2 Mini UD-IQ3_XXS, SSD | 80.88 s | 126.39 s | 1 + 1 | 4,262 MiB | Both turns and regression passed |

| Model and request | Evaluated / cached input | Output | Prefill | Decode | First text |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemma, turn 1 | 773 / 0 | 324 | 151.02 tok/s | 17.71 tok/s | 5.30 s |
| Gemma, turn 2 first attempt | 1,003 / 768 | 295 | 143.78 tok/s | 15.46 tok/s | 7.18 s |
| Gemma, turn 2 retry | 373 / 1,767 | 439 | 106.30 tok/s | 15.53 tok/s | 3.77 s |
| Inkling IQ2, turn 1 | 856 / 0 | 228 | 67.09 tok/s | 4.61 tok/s | 12.77 s |
| Inkling IQ2, turn 2 | 793 / 1,083 | 369 | 64.64 tok/s | 4.42 tok/s | 12.28 s |

Inkling implemented the integer-cent tax and total, forwarded the tax into the cost summary, applied credit after tax, and reused the invoice function in the credit cost summary. Gemma corrected the credit cost summary after its first turn-2 test failed. These results measure one pass per model on this fixture and controller.

```bash
./download.sh iq2
INKLING_PORT=18080 INKLING_REASONING_BUDGET=0 ./serve.sh coding
```

In another terminal, with PHP installed on macOS:

```bash
probe_dir=$(mktemp -d)
server_pid=$(lsof -tiTCP:18080 -sTCP:LISTEN)
python3 -B tools/php_patch_probe.py "$probe_dir/inkling" \
  --model inkling-2-mini --pid "$server_pid" --footprint ./build/bin/inkling-footprint \
  --attempts 2 --max-tokens 512 --prefill-json
```

For Gemma, start the same server build with `INKLING_CTX=8192 INKLING_BATCH=512 INKLING_PORT=18080 ./serve.sh resident /path/to/gemma-4-12B-it-Q4_0.gguf`, then run the probe without `--prefill-json`. Stop each server before starting the next. [Gemma checkpoint and native-tool baseline](gemma-4.md)
