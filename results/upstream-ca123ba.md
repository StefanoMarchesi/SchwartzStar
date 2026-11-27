# The patch series on a newer llama.cpp (ca123ba)

2026-11-27, MacBook with M1 Pro and 16 GB, Inkling 2 Mini `IQ2_XXS`. This is a compatibility check. `build.sh` still pins `ae6a7ad`.

## Build

On upstream llama.cpp `ca123ba76311290ec6ab041dd2058fed8d9f03d2` (2026-11-26), patches `0001`-`0010` apply in order with `git apply`, and `llama-server` and `test-backend-ops` build without warnings. The test program was built, not run. Patch `0011` (NVFP4, banded series) does not apply; upstream has its own NVFP4 code, which was not tested.

The banded attention path is still needed for long context. Upstream now passes Inkling's relative-position bias to flash attention as an indexed table, but at this revision the Metal backend reports that form as unsupported and falls back to the dense mask.

## SSD streaming against resident weights

Three prompts of about 1,900 tokens, 128 greedy tokens, K and V `q4_0`, expert cache 3072 MiB. In the pinned build and in the newer one, SSD streaming and resident weights gave the same 128 tokens and the same log-probabilities.

## Pinned build against newer build

The two builds do **not** give the same output, with resident weights too. On those prompts the text diverged after 95 to 100 tokens, and the first-token log-probability differed by 0.10 to 0.92.

To see where this comes from:

- On the CPU (no flash attention, F16 KV, prompts of about 300 tokens, three prompts, 16 tokens) the two builds stayed close: mean log-probability difference 0.005 to 0.011. The rewritten model graph computes nearly the same thing.
- On Metal with the usual settings, eight prompts of about 300 tokens were compared with a CPU run of the newer build (F16 KV). Over the top-5 tokens in common at the first step, the mean absolute log-probability difference was 0.265 for the pinned build and 0.260 for the newer one. Each picked the same first token as the CPU on 7 of 8 prompts. This distance also includes the effect of `q4_0` KV and flash attention.
- Turning Metal graph fusion off changed the text on 4 of 8 prompts in the pinned build and 1 of 8 in the newer one.

On this small sample neither build is closer to the CPU reference. Small rounding differences on Metal are enough to change greedy text with this quantized model, so exact output cannot be compared across llama.cpp revisions.

## Recall and speed

Code recall at about 7,900 tokens: 8 of 8 for both builds on two documents. Peak process memory: 3,461 and 3,463 MiB pinned, 3,474 and 3,477 MiB newer.

SSD streaming, one prompt of 1,887 tokens, four pairs in alternated order, pinned to newer:

| Pair | Decode tok/s | Prompt tok/s |
| --- | --- | --- |
| 1 | 6.00 → 4.98 | 63.2 → 54.0 |
| 2 | 7.68 → 7.31 | 64.6 → 65.1 |
| 3 | 6.71 → 7.47 | 65.1 → 67.0 |
| 4 | 7.61 → 7.47 | 63.4 → 67.3 |

No consistent speed difference. The Mac was in use and background load changed between runs. Because the texts differ, the newer build stopped at 194 generated tokens and the pinned one at 256, so the decode rates cover different lengths.

Not checked: long context, the experimental patches (keep-alive, in-place I/O, Turbo3) and speculative decoding on the newer revision.
