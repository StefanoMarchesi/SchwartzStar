# Gemma 4 12B Instruct Q4_0 — PHP fixture reference

Profile reconstructed from the historical launcher: `llama-server`, Metal with `--n-gpu-layers 99`, Jinja, one sequence, context 4.096. The launcher uses the runtime's default KV (F16/F16 in the verified binary). For this repeat, the Homebrew binary returned `b10964-b29c606e2`. The version from the old run remains unknown.

The historical checkpoint is unavailable. The repeat uses the same-named GGUF published by ggml-org, revision `501d13dba70e3c02b1b12c78fd9e6b08ad413960`, 7.219.673.216 bytes, verified SHA-256 `3712b9bd32cae83a22f67ee7a4466d8d7a4f21646ac8a07d19bf9418e8767a70`. No hash is available for the historical file for comparison. [Pinned GGUF](https://huggingface.co/ggml-org/gemma-4-12B-it-GGUF/resolve/501d13dba70e3c02b1b12c78fd9e6b08ad413960/gemma-4-12B-it-Q4_0.gguf)

## Commands

```bash
llama-server --model "./models/gemma-4-12B-it-Q4_0.gguf" \
  --host 127.0.0.1 --port 18080 --no-webui --ctx-size 4096 \
  --parallel 1 --n-gpu-layers 99 --jinja
```

From another terminal, use a new directory for every test:

```bash
pid=$(lsof -tiTCP:18080 -sTCP:LISTEN)
model_id=$(curl -fsS http://127.0.0.1:18080/v1/models | \
  python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
probe_dir=$(mktemp -d "${TMPDIR:-/tmp}/gemma-php.XXXXXX")
python3 -B tools/php_agent_probe.py "$probe_dir/standard" --model "$model_id" \
  --pid "$pid" --footprint "./build/bin/inkling-footprint"
python3 -B tools/php_agent_probe.py "$probe_dir/budget-zero" --model "$model_id" \
  --pid "$pid" --footprint "./build/bin/inkling-footprint" --reasoning-budget 0
python3 -B tools/php_agent_probe.py "$probe_dir/no-thinking" --model "$model_id" \
  --pid "$pid" --footprint "./build/bin/inkling-footprint" \
  --reasoning-budget 0 --disable-thinking
```

## First attempts on the Mac

| Configuration | Total API input | Re-evaluated / reused | First tool call | First text | Total time | Footprint at end of turn | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Historical launcher, default reasoning | 490 | 490 / 0 | — | — | 53,95 s | 948,5 MiB | Exhausted 768 reasoning tokens, initial test failed |
| Same model, zero reasoning budget | 3.557 | 1.178 / 2.379 | 4,56 s | — | 66,09 s | 3.168,6 MiB | Invoice calculation correct, cost summary still wrong |

First attempt: 106,87 tok/s prefill, 15,65 tok/s decode, 515,7 MiB swap increase. Zero budget: four requests, with 485, 857, and 1.037 reused tokens in the three passes after the first. Process physical-footprint peak of about 3,1 GiB, while system swap rose from 3.639,1 to 6.149,6 MiB. macOS pressure level 2 at the end of both turns. The footprint is the lifetime peak reported by macOS. The machine was not under stable load and the timings are single-session measurements.

## Gemma with zero budget and template reasoning disabled

Same fixture, prompts, and runtime. The first request starts with a cold prefix. Subsequent passes use ordinary server reuse.

| Turn | Requests measured | Total API input | Re-evaluated / reused | First tool call | First text | Time | Test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1, invoice and costs | 9 | 13.873 | 2.385 / 11.488 | 17,41 s | 103,81 s | 108,85 s | Green at 102,74 s, both source files modified |
| 2, credit | 5 completed + 1 HTTP 500 | 16.276 partial | 1.609 / 14.667 partial | 3,35 s | — | at least 53,01 s | Failed, credit API missing and regression on the first test |

Tokens sum across successive requests, so part of the context is counted more than once. The first warm pass of turn 2 processed **239 tokens out of 2.614**, reusing **2.375**. Aggregate decode was **16,24 tok/s** in turn 1 and **16,89 tok/s** in the five measured requests of turn 2. Maximum footprint was **8.898,3 MiB** at the end of turn 1 and **12.152,8 MiB** before stopping. RSS at the end of turn 1 was **581,8 MiB**. System swap was **5.805,6 → 11.974,8 MiB** in turn 1 and approximately **15.429 MiB** after the error. macOS-reported free percentage was **19 → 13%** in turn 1, pressure level **2**. System-memory measurements include the Mac's workload.

The fifth request of turn 2 started with **3.980 tokens** and stopped at **4.095 occupied positions**. The replacement arguments remained incomplete. The sixth request returned HTTP 500 for unclosed JSON arguments and the client exited with code 1. Independent checking of the final state returned error 255 for missing `invoice_total_cents()` and exit 1 with `credit API is missing`. The turn-1 diff and JSON preserve the earlier green result.

### Verified final no-thinking baseline

The final verified no-thinking baseline from `HANDOFF-INKLING-128K.md` is turn 1 green at **102,74 s** for the first text and **108,85 s** total. Turn 2 reached the 4K truncation, returned HTTP 500, and was followed by a regression on turn 1. The fixture contract was adjusted in subsequent probes. The historical Wiki records build **10310**, commit `cb26014d9`; that build is separate from this repetition.

Private raw artifacts from the run were retained outside the repository; they contain the requests, SSE events, fixture, diff, metrics, errors, and timing. The Gemma process was closed after collection.
