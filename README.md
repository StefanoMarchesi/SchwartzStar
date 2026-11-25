# SchwartzStar

[![Yogurt in Spaceballs saying “May the Schwartz be with you!”](https://media.giphy.com/media/oyM0l7D6f7gdUKcn41/giphy.gif)](https://giphy.com/gifs/yogurt-spaceballs-may-the-schwartz-be-with-you-oyM0l7D6f7gdUKcn41)

> “I see your Schwartz is as big as mine!”
>
> 16 GB of unified RAM. A 17.63 GB GGUF on SSD. [The *Spaceballs* scene](https://www.youtube.com/watch?v=cBdeF14G1VM)

SchwartzStar runs **Thinking Machines Lab’s Inkling 2 Mini** on Apple Silicon by streaming routed expert weights from SSD into a bounded Metal cache. It ships as a patch set for `llama.cpp` and was tested on a 16 GB MacBook Pro with an M1 Pro. The name is a nod to *Spaceballs* and to Salvatore Sanfilippo’s [DwarfStar](https://github.com/antirez/ds4/blob/main/docs/SSD_STREAMING.md), which inspired the SSD streaming design.

## Quick start

You need Apple Silicon, Git, CMake, Python 3, and Xcode with the Metal toolchain. Leave room on the internal SSD for a GGUF, the build, and macOS. `download.sh` verifies the pinned SHA-256 and resumes partial downloads.

```bash
git clone https://github.com/StefanoMarchesi/SchwartzStar.git
cd SchwartzStar
./build.sh
MODEL=$(./download.sh iq2)
./serve.sh coding "$MODEL"
```

The server listens on `127.0.0.1:8080` and advertises `inkling-2-mini`. A GGUF supplied by path must use the current `inkling2_loop` layout. Use a minimal prompt and tool set for interactive work. The `coding` profile enables a 128-token reasoning budget by default; allow room for reasoning and the final answer in the client's output limit.

```bash
curl -fsS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"inkling-2-mini","messages":[{"role":"user","content":"Write a Python function that clamps a number to a range."}],"max_tokens":1024,"stream":true}'
```

| Profile | Default GGUF | Expert cache | Context | Microbatch |
| --- | --- | ---: | ---: | ---: |
| `coding` | Unsloth UD-IQ2_XXS, 8.49 GB | 3,072 MiB SSD | 8,192 | 512 |
| `quality` | ggml-org IQ4_NL, 17.63 GB | 768 MiB SSD | 1,024 | 2 |
| `minimum` | Unsloth UD-IQ1_S, 7.76 GB | 512 MiB SSD | 1,024 | 1 |
| `resident` | Unsloth UD-IQ2_XXS, 8.49 GB | Resident weights | 2,048 | 16 |

`INKLING_CTX`, `INKLING_BATCH`, `INKLING_CACHE_MIB`, `INKLING_THREADS`, and `INKLING_PORT` override the profile. `INKLING_CACHE_TYPE_K` defaults to `f16`. `q4_0` is an experimental option for the key cache. The value cache stays F16. The server reserves enough expert slots for each microbatch and reports the required cache size if the settings cannot fit. `INKLING_CTX_CHECKPOINTS` controls the number of in-memory prefix checkpoints and defaults to two.

The default 8K context is suitable for short prompts, not established as sufficient for repository-scale agent tasks. In one SWE-bench Pro Hard pilot, a request reached 20,546 tokens and exceeded a 16,384-position limit before a patch was submitted. A 131,072-position configuration processed more than 22,000 tokens before the pilot was stopped, but neither run produced a score. Increasing the configured context also increases memory use; a large allocation alone does not demonstrate that a fully populated context is usable. [Pilot details](results/banded-ssd-pilot.md)

To experiment with the 128K allocation used in that pilot, start the server with:

```bash
INKLING_CTX=131072 INKLING_BATCH=64 INKLING_CACHE_MIB=2304 \
  INKLING_CACHE_TYPE_K=q4_0 ./serve.sh coding "$MODEL"
```

This configuration uses experimental quantized keys and a smaller microbatch. It can be substantially slower and put more pressure on memory than the default. Full 128K retrieval and coding quality, including tool calls, remain unverified; monitor memory and swap on your own machine.

The `coding` profile defaults to minimal reasoning with a 128-token budget; set `INKLING_REASONING_BUDGET` to change it, or to `0` to turn reasoning off. The other profiles default to `0`. A budget of 128 recovered visible code on all three tasks of the initial EvalPlus instruct pilot in resident mode. This small pilot does not establish the best budget for complex work. [Reasoning pilot](results/humaneval-smoke.md#official-instruct-prompt-pilot-with-reasoning-enabled)

To connect a coding client, see the [mini-SWE-agent setup](docs/mini-swe-agent.md), including its host execution and context limitations.

An experimental [single-task DeepSWE runner](tools/deepswe-pilot/README.md) is available for a local server and Pier. Its workflow has passed a simulated end-to-end preflight; a real DeepSWE model score is not yet available.

## Measurements on a 16 GB M1 Pro

| Model and configuration | Input / context | Prefill | Decode | First text | Peak process footprint |
| --- | ---: | ---: | ---: | ---: | ---: |
| IQ2 SSD, u512, 3,072 MiB expert cache | 512 / 4,096 | 87.26 tok/s | 4.60 tok/s | 5.87 s | 3,666 MiB |
| IQ3 SSD, u512, 3,072 MiB expert cache | 512 / 4,096 | 66.44 tok/s | 3.42 tok/s | 7.71 s | 3,638 MiB |
| IQ4 SSD with Q4_0 keys, u1, 256 MiB expert cache | 32 / 131,072 | 2.17 tok/s | 2.28 tok/s | 14.76 s | 3,175 MiB |

These are single runs on the internal SSD. The 128K row measures startup and short inference with a 131,072-position allocation. Long-context quality depends on testing a populated prefix. Process physical footprint, RSS, Metal allocations, and system swap are separate measurements. See the [measurement report](results/schwartzstar.md), [long-context notes](LONG-CONTEXT.md), and [Gemma 4 baseline](results/gemma-4.md).

```bash
MODEL=$(./download.sh iq2)
INKLING_CTX=4096 INKLING_REASONING_BUDGET=0 python3 check.py run coding "$MODEL" iq2-ssd-p512 \
  --prefix-only 512 --tokens 24 --timeout 180
```

The probe measures evaluated and generated tokens, first visible text, Metal I/O coverage, peak memory, and system swap. IQ2 SSD passed both turns and the regression check in the [guided coding benchmark](results/coding-agent.md), with 4,290 MiB peak process footprint. Turn times were 62.19 s and 95.71 s. Resident Gemma 4 12B Q4_0 passed the same tests in 23.55 s and 58.51 s. These times come from a synthetic PHP fixture and a controller that applies structured edits.

In a separate retrieval check, IQ2 recovered all three records in an 8,000-token prompt with Q4_0 keys. Prefill was 67.75 tok/s and peak process footprint was 3,137 MiB. [Retrieval setup and output](results/schwartzstar.md#8000-token-retrieval-probe)

At 16,000 evaluated tokens in the same raw completion test, IQ2 again recovered all three codes and peaked at 4,109 MiB RSS. Resident Gemma returned three incorrect codes and used 8,024 MiB RSS. Gemma began answering in 118 seconds, versus 267 seconds for IQ2. The result covers one constrained retrieval test with Q4_0 keys. [Configuration and memory metrics](results/schwartzstar.md#comparison-with-resident-gemma-at-8k-and-16k)

On the first three tasks from HumanEval+, Inkling IQ2 passed 2/3 with an explicit assistant text prefix. Gemma passed 3/3. This smoke test uses the official EvalPlus tests for those tasks. [Task-level results and reproduction](results/humaneval-smoke.md)

Both models also passed all 19 tests on one guided Flask case from SWE-bench Lite after receiving failure feedback. Gemma finished sooner. Inkling solved the case with one new API parameter. [Flask case and protocol](results/swebench-lite-smoke.md)

In a separate mini-SWE-agent smoke test, Inkling fixed a small shell script in both runs while Gemma exhausted the tool-call limit without a patch. The order was reversed for the repeat. [Fixture and results](results/mini-swe-agent-smoke.md)

An isolated, unreleased Metal candidate improved prefill in short SSD streaming comparisons while producing matching output. These measurements do not establish coding quality or full 128K context use. [Experimental results and limitations](results/banded-ssd-pilot.md)

## How it works

- Inkling’s twelve central layers run three times. The 24 physical layers yield 48 applications, each with its own state. `adishik`’s shared expert cache work lets repeated applications reuse the same cached weights. [Thinking Machines Lab](https://thinkingmachines.ai/news/inkling-2-mini/) · [Upstream model support](https://github.com/ggml-org/llama.cpp/pull/32694) · [Expert cache deduplication](https://github.com/ggml-org/llama.cpp/issues/32744)
- The model keeps expert tensors in a cold file mapping. After the Metal router returns the selected IDs, bounded workers `pread` cache misses directly into shared CPU/GPU buffers using `F_NOCACHE`. The runtime verifies the allocation and counts all 48 layer applications. [DwarfStar SSD streaming](https://github.com/antirez/ds4/blob/main/docs/SSD_STREAMING.md) · [Alex Schiltmans’s per-layer expert streaming](https://github.com/alexschiltmans/llama.cpp-per-layer-experts/blob/per-layer-experts/docs/moe-ssd-streaming.md)
- The runtime budgets by quantization layout and reserves enough slots for a full microbatch. Prefill can group more tokens before the next router readback. The benchmark measures the effect independently of generated token speed. [Unsloth GGUFs](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) · [ggml-org IQ4_NL](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF)

## Hardware scope

The SSD path uses Apple Silicon shared Metal buffers. A lower-memory Mac would need a smaller expert cache and checks for swap and system pressure with its usual apps open. Upstream Inkling builds and systems such as [Maya Inkling Mini](https://github.com/alesha-pro/maya-inkling-mini/blob/main/docs/INKLING2-MINI.md) already describe GPU/RAM offload on discrete GPUs. A SchwartzStar SSD port for NVIDIA or AMD would need a bounded host staging pool, asynchronous PCIe copies, and separate VRAM and system memory budgets. [Unsloth backend documentation](https://unsloth.ai/docs/models/inkling-2-mini)

[License](LICENSE) · [Revisions, checkpoints, and release notes](RELEASE-v0.1.0-alpha.md)
