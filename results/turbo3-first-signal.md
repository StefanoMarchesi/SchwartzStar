# Turbo3 value cache: a first signal

Apple M1 Pro, 16 GB, 26 November 2026. This is a short screening test, not a quality result.

**Correction, same day.** An earlier version said turbo3 with `q4_0` keys ran on the Metal flash attention path. That was wrong. In that tree Metal accepts different key and value types only for turbo with turbo or `q8_0` with turbo. For `q4_0`/`f16`, `q4_0`/`q8_0` and `q4_0`/`turbo3` the attention fell back to the CPU: those runs had 194 graph splits instead of 98 and a larger CPU compute buffer. The only turbo3 run on Metal here is `q8_0`/`turbo3`.

## Setup

An experimental llama.cpp tree that includes TurboQuant types, with local changes for SSD expert streaming on Metal. It is not the build in this repository, and this repository cannot reproduce it. Flash attention on, 4,096 positions, microbatch 8 (a limit of that tree), 2,304 MiB expert cache, greedy sampling. The experimental banded path was not used. The backend of each configuration is inferred from the Metal support rules in the code, the graph split count and the compute buffer sizes; nodes were not traced one by one.

Two synthetic documents of about 1,900 tokens, each with eight planted codes, asked back under a JSON schema.

## Results

All six configurations returned 8 of 8 codes on both documents, with the same text as `f16`/`f16`. All 186 generated tokens matched, so log-probabilities can be compared step by step.

| K / V | Attention ran on | Mean absolute log-probability difference from f16/f16 | Largest | KV cache MiB (8 full layers + 40 sliding-window layers) |
| --- | --- | ---: | ---: | ---: |
| q4_0 / f16 | CPU | 0.0095 | 0.805 | 82.0 + 128.1 |
| q4_0 / q8_0 | CPU | 0.0075 | 0.536 | 52.0 + 81.3 |
| q4_0 / q4_0 | Metal | 0.0114 | 0.762 | 36.0 + 56.3 |
| q4_0 / turbo3 | CPU | 0.0093 | 0.528 | 30.5 + 47.7 |
| q8_0 / turbo3 | Metal | 0.0228 | 1.230 | 46.5 + 72.7 |

The value part of the turbo3 cache took 400 bytes per cell and head group, against 576 for `q4_0`.

## What this does and does not show

- With `q4_0` keys, turbo3 values stayed as close to `f16` as the other formats on these short prompts, with attention on the CPU. This says nothing about a Metal kernel for that pair, which that tree does not have on the dense path.
- On Metal, `q8_0` keys with turbo3 values returned the same tokens, with the largest mean difference of the table.
- The test hit its ceiling: no format made an error. The differences between formats are small and come from two documents.
- It says nothing about long context. A [third-party CUDA test](https://sotaaz.com/post/turboquant-llamacpp-bench-en) on another model and fork reported turbo3 for both K and V degrading badly from 8K tokens up, while `q8_0` keys with turbo3 values stayed close to `f16`.
- Speed is not meaningful here: this tree ran prompts at about 3 to 5 tok/s with either type.
- The SSD build in this repository has no turbo3 types. Testing it at long context needs a port.

## Memory at long context

Computed from the block sizes, for the 8 full layers only. With `q4_0` keys, turbo3 values would take 976 MiB at 131,072 positions against 1,152 MiB for `q4_0` values and 1,664 MiB for `q8_0` values. The Metal compute buffer is a comparable or larger cost at that size ([measurements](banded-long-context.md)).
