# Turbo3 value cache on the banded build

Apple M1 Pro, 16 GB, 26 November 2026. Experimental: not part of `build.sh` or any profile.

## Update: turbo3 in banded attention

27 November 2026. `patches-experimental/turbo3-banded.patch` applies on top of `turbo3-v-cache.patch` and lets banded flash attention run on Metal with `q4_0` keys and turbo3 values at head size 128. It changes the Metal support check and adds 29 test cases; the existing kernels already handled the turbo3 block. Cell positions and K/V row lists stay unsupported there.

Checks, rebuilt separately from the author's build: FLASH_ATTN_EXT_BANDED 321 passed (29 with turbo3), FLASH_ATTN_EXT 5,428, SET_ROWS 174, TURBO_WHT 3, none failed. A verbose run showed that the patched build compiles the banded turbo3 kernels (`rel=1`, `kvpad=1`) and the dense build does not.

At 7,895 prompt tokens it recalled 8/8 codes with the same 94 output tokens and text as the dense turbo3 build and as `f16` values. At 31,883 tokens (microbatch 512, 3,072 MiB expert cache, one run per setting, same document as the earlier 32K runs, which were measured in an earlier session):

| V cache and attention | Codes | Peak process footprint, MiB | Prefill tok/s |
| --- | --- | ---: | ---: |
| turbo3, banded | 8/8 | 3,617 | 52.0 |
| turbo3, dense | 8/8 | 4,001 | 52.7 |
| `q4_0`, banded (the `long` profile) | 8/8 | 3,668 | 68.1 |
| `f16`, banded | 8/8 | 4,124 | 75.1 |

All four wrote the same text. Mean absolute log-probability difference from `f16` values was 0.016 for banded turbo3 and 0.018 for `q4_0`. A second document at 31,888 tokens also gave 8/8.

So banded attention removes the extra peak memory of the dense turbo3 path (about 384 MiB here), and at 32K turbo3 uses about 50 MiB less than `q4_0` values. It did not make prompt processing faster: turbo3 stays about a quarter slower than `q4_0` there. The speeds are single runs. The Metal compute buffer reported at load was the same for the dense and banded builds, so it does not show this difference. Longer contexts were not tested.

The patch was written with [Freebuff](https://freebuff.com/?ref=ref-b181061b-0b36-403c-9e01-a696e0b1b6fc&via=rewards_link) running DeepSeek V4.1 Flash, then reviewed and checked with the model with another AI assistant. Disclosure: I am not affiliated with Freebuff. That is my invite link, and Freebuff's programme may reward me if you use it; as far as I know this is only its points collection for merchandise, and the app also offers usage hours for invitations.

## What was ported

`patches-experimental/turbo3-v-cache.patch` applies on top of the banded build sources (26 files). It adds the turbo3 type (400 bytes per 1,024 values), its CPU codec, the cache write path and the inverse rotation, plus Metal kernels for cache writes, the rotation and dense flash attention with `q4_0` keys and turbo3 values. Turbo3 keys and other key types are not supported. This first patch ran turbo3 only in dense attention; see the update above for banded attention.

## Checks

- The ported CPU quantizer produced the same bytes as the original on 360 rows (126,000 bytes).
- With `q4_0`/`q4_0`, the patched build gave the same tokens and log-probabilities as the unpatched build on a 7,895-token prompt (measured before the Metal kernels were added).
- With attention on the CPU, turbo3 against `f16` deviated about as much as the original implementation on the same two documents.
- `test-backend-ops` on Metal, compared with the CPU: SET_ROWS 174 passed (3 with turbo3), TURBO_WHT 3 passed, FLASH_ATTN_EXT 5,428 passed (2 with turbo3), none failed. The turbo3 attention cases do not cover attention sinks, which the model uses; the model runs below do.
- In the model runs, the turbo3 attention stayed on Metal: same graph splits and CPU compute buffer as `f16`, and the cache was in Metal buffers.

## Recall and output

Synthetic documents with eight planted codes, JSON schema, greedy, 3,072 MiB expert cache, K `q4_0`.

| Prompt tokens | Documents | turbo3 codes | Same tokens as f16 V | Mean absolute log-probability difference from f16 V |
| ---: | ---: | --- | --- | --- |
| about 1,900 | 2 | 16/16 | 186/186 | 0.017 and 0.021 |
| 7,895 | 1 | 8/8 | 94/94 | 0.025 |
| 31,883 | 1 | 8/8 | 92/92 | 0.013 (`q4_0` V in an earlier run: 0.018) |

The recall test has hit its ceiling in earlier runs: every cache format passed. It does not rank formats and says nothing about coding or agent work.

## Memory

Metal KV cache at 32,768 positions: 301 MiB with turbo3 V against 810 MiB with `f16` V. Against `q4_0` V, the current `long` profile, the saving is about 54 MiB at this size. From the block sizes, the full-context layers would save about 176 MiB at 131,072 positions. The compute buffer did not change.

## Speed

Turbo3 was slower in prompt processing.

- At 7,895 tokens, microbatch 512, mirrored order (f16, turbo3, q4_0, q4_0, turbo3, f16), means of two runs:

| V cache | Prefill tok/s | Decode tok/s |
| --- | ---: | ---: |
| f16 | 87.8 | 4.27 |
| q4_0 | 82.6 | 5.00 |
| turbo3 | 69.7 | 3.74 |

  Turbo3 prefill was about 16% below `q4_0` and 21% below `f16`. Decode varied too much between repeats of the same setting (3.5 to 6.5 tok/s for `q4_0`) to compare.
- At 31,883 tokens, one pair: 52.7 against 75.1 tok/s prefill and 3.93 against 4.50 tok/s decode, turbo3 against `f16`.

## Conclusion

Turbo3 with `q4_0` keys works on Metal and kept the recall results up to 32K. At these sizes it saves little memory against `q4_0` values and processes prompts more slowly, so the `long` profile stays on `q4_0`. It may matter at much longer contexts, which were not tested.

## Reproducing

Build the banded variant first, then patch a copy of its sources. These commands were not run in this exact form; the measured build used the same sources.

```bash
INKLING_VARIANT=banded ./build.sh
cp -R .llama.cpp-banded .llama.cpp-turbo3
git -C .llama.cpp-turbo3 apply "$PWD/patches-experimental/turbo3-v-cache.patch"
git -C .llama.cpp-turbo3 apply "$PWD/patches-experimental/turbo3-banded.patch"
cmake -S .llama.cpp-turbo3 -B build/turbo3 -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
cmake --build build/turbo3 --target llama-server --parallel 4
```

The measurements started the server with the same flags as the `long` profile apart from `--cache-type-v turbo3` and no checkpoints. Starting it through `serve.sh` was not tested:

```bash
INKLING_SERVER=build/turbo3/bin/llama-server INKLING_CACHE_TYPE_V=turbo3 ./serve.sh long "$(./download.sh iq2)"
```
