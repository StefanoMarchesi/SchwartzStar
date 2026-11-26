# Turbo3 value cache: a first signal

Apple M1 Pro, 16 GB, 26 November 2026. This is a short screening test, not a quality result.

## Setup

An experimental llama.cpp tree that includes TurboQuant types, with local changes for SSD expert streaming on Metal. It is not the build in this repository, and this repository cannot reproduce it. Flash attention on, 4,096 positions, microbatch 8 (a limit of that tree), 2,304 MiB expert cache, greedy sampling. Turbo3 ran on the ordinary dense flash attention path, not on the experimental banded path. Which backend executed each attention node was not traced.

Two synthetic documents of about 1,900 tokens, each with eight planted codes, asked back under a JSON schema.

## Results

All six configurations returned 8 of 8 codes on both documents, with the same text as `f16`/`f16`. All 186 generated tokens matched, so log-probabilities can be compared step by step.

| K / V | Mean absolute log-probability difference from f16/f16 | Largest | KV cache MiB (8 full layers + 40 sliding-window layers) |
| --- | ---: | ---: | ---: |
| q4_0 / f16 | 0.0095 | 0.805 | 82.0 + 128.1 |
| q4_0 / q8_0 | 0.0075 | 0.536 | 52.0 + 81.3 |
| q4_0 / q4_0 | 0.0114 | 0.762 | 36.0 + 56.3 |
| q4_0 / turbo3 | 0.0093 | 0.528 | 30.5 + 47.7 |
| q8_0 / turbo3 | 0.0228 | 1.230 | 46.5 + 72.7 |

The value part of the turbo3 cache took 400 bytes per cell and head group, against 576 for `q4_0`.

## What this does and does not show

- With `q4_0` keys, turbo3 values loaded, ran and stayed as close to `f16` as the other formats on these short prompts.
- The test hit its ceiling: no format made an error. The differences between formats are small and come from two documents.
- It says nothing about long context. A [third-party CUDA test](https://sotaaz.com/post/turboquant-llamacpp-bench-en) on another model and fork reported turbo3 for both K and V degrading badly from 8K tokens up, while `q8_0` keys with turbo3 values stayed close to `f16`.
- Speed is not meaningful here: this tree ran prompts at about 3 to 5 tok/s with either type.
- The SSD build in this repository has no turbo3 types. Testing it at long context needs a port.

## Memory at long context

Computed from the block sizes, for the 8 full layers only. With `q4_0` keys, turbo3 values would take 976 MiB at 131,072 positions against 1,152 MiB for `q4_0` values and 1,664 MiB for `q8_0` values. The Metal compute buffer is a comparable or larger cost at that size ([measurements](banded-long-context.md)).
