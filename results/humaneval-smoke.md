# HumanEval+ smoke test on a 16 GB M1 Pro

The first three tasks of HumanEval+ v0.1.10 were run through EvalPlus 0.3.1, using both the original and expanded test inputs. Each model generated one greedy completion per task, with a cap of 256 output tokens. These results cover three Python function completions. A full-suite score and repository-level agent evaluation require separate runs. [EvalPlus](https://github.com/evalplus/evalplus) · [HumanEval+ release](https://github.com/evalplus/humanevalplus_release/releases/tag/v0.1.10)

| Task | Inkling IQ2, streamed, text prefix | Gemma 4 12B Q4_0, resident |
| --- | --- | --- |
| `HumanEval/0` | Pass | Pass |
| `HumanEval/1` | Fail | Pass |
| `HumanEval/2` | Pass | Pass |
| Original tests | 2/3 | 3/3 |
| Expanded tests | 2/3 | 3/3 |

Both servers used Metal, four threads, one slot, 8,192 context positions, microbatch 512, and two context checkpoints. The Inkling `coding` profile used UD-IQ2_XXS and a 3,072 MiB expert cache. Gemma used the `resident` profile. Without the text prefix, the Inkling server returned no visible code for all three completions within the token cap. The scored run supplied the indented comment `# Solution` as the start of the assistant text block, with `echo` enabled. Gemma used the default completion path. The protocol therefore measures each setup with its documented response handling.

| Task | Inkling request wall time | Gemma request wall time |
| --- | ---: | ---: |
| `HumanEval/0` | 18.3 s | 4.2 s |
| `HumanEval/1` | 22.5 s | 7.0 s |
| `HumanEval/2` | 5.7 s | 1.5 s |

The recorded run scored the completions with the official evaluator APIs in a restricted macOS process with networking disabled and CPU and wall-time limits. The evaluator's address-space limit was disabled for macOS compatibility. The reproduction commands print a Docker command for isolated evaluation.

## Reproduce

Start the [coding profile](../README.md#quick-start), then generate the fixed subset in another terminal.

```bash
run_dir=$(mktemp -d)
python3 tools/evalplus_smoke.py "$run_dir/inkling" \
  --port 8080 --model inkling-2-mini --max-tokens 256 --prefill-text
```

Stop the Inkling server. Start the same build with the [Gemma baseline checkpoint](gemma-4.md) and run the probe with the server's advertised alias.

```bash
INKLING_CTX=8192 INKLING_BATCH=512 ./serve.sh resident "$GEMMA_MODEL"
```

```bash
python3 tools/evalplus_smoke.py "$run_dir/gemma" \
  --port 8080 --model inkling-2-mini --max-tokens 256
```

Run the printed Docker command for each output directory. It mounts that run directory and the scorer script and disables container network access. `evalplus-smoke-results.json` contains the task-level statuses. The dataset used here has SHA-256 `272720b90ac375502c8ed23cd791c2a93dfb22a911641a494da74a426c09f101`.

## Official instruct-prompt pilot with reasoning enabled

The three disclosed tasks were also run through EvalPlus 0.3.1's OpenAI-compatible instruct generator and official sanitizer. Both models used the resident profile, Metal, one slot, four threads, 8,192 context positions, and microbatch 512. Each task used one greedy response capped at 768 generated tokens. No assistant prefix was added in this protocol.

| Model and server setting | Original tests | Expanded tests | Median server inference time |
| --- | ---: | ---: | ---: |
| Gemma 4 12B Q4_0, default | 3/3 | 3/3 | 20.88 s |
| Inkling IQ2, reasoning off, zero budget | 0/3 | 0/3 | 55.11 s |
| Inkling IQ2, reasoning on, minimal effort, 128-token budget | 3/3 | 3/3 | 28.29 s |

The zero-budget configuration repeatedly opened and closed an empty thinking block and reached the 768-token cap for all three tasks. With reasoning on, the model produced visible code and stopped after 379, 437, and 263 generated tokens. This pilot uses the same official tasks as the completion experiment above, with a different prompt protocol and Inkling runtime profile. [EvalPlus generation protocol](https://github.com/evalplus/evalplus/blob/master/docs/cli.md)

## Reasoning-budget sweep

The same three tasks were then issued with the same official instruct prompt, greedy decoder, and 768-token output cap. One resident IQ2 server at 8,192 configured positions handled all nine requests. The requests used streaming so the first visible text could be timed. Prefix caching was disabled, and the order of budgets was rotated between tasks. EvalPlus sanitized and evaluated the returned code.

| Thinking budget | HumanEval+ tests passed | Median time to first visible text | Median request time |
| ---: | ---: | ---: | ---: |
| 32 | 3/3 | 4.86 s | 19.51 s |
| 128 | 3/3 | 11.38 s | 28.01 s |
| 256 | 3/3 | 19.35 s | 30.65 s |

The 32-token profile passes the same three tasks with the lowest latency in this sweep. The results are confined to this sample. The configured capacity was 8,192 tokens and the three prompt lengths were under 200 tokens. Populating longer contexts is a separate test.
