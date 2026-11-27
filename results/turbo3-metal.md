# Turbo3 value cache on the banded build

Apple M1 Pro, 16 GB, 26 November 2026. Experimental: not part of `build.sh` or any profile.

## Update: a populated 256K prompt

27 November 2026. One run: turbo3 values, K `q4_0`, 262,144 context, **microbatch 256** (chosen to save memory), 3,072 MiB expert cache, banded prompt processing, all the turbo3 patches below, 16 GB M1 Pro.

| | Value |
| --- | --- |
| Prompt tokens | 260,996 |
| Planted codes recalled | 8/8 |
| Prefill | 39.6 tok/s, 1 h 50 min |
| Decode, 92 tokens | 3.0 tok/s |
| Peak process footprint | 5,523 MiB |
| Swap growth, sampled every 15 s | up to 4.6 GB (4.4 GB at the end) |
| Lowest free memory reported by macOS | 8% |
| SSD reads in prefill | 1,312 s (20% of prefill), 4,557 GiB |
| Expert cache hit rate in prefill / decode | 42.5% / 90.2% |

- Prompt processing slowed along the prompt: about 50 tok/s up to 32K, 44 tok/s between 65K and 131K and 32 tok/s over the last 64K. Part of the slowdown may come from the memory pressure; the causes were not separated.
- **It completed, but not comfortably.** Swap grew heavily and free memory was very low. The run stopped short of the 6 GB swap-growth limit set to abort it.
- The machine was busy during part of this run: other AI agents, a 648 MB download, desktop apps, and an external disk that disconnected (the test files were on the internal drive). Speeds are not a clean measurement.
- Reads per prompt token were about 17.9 MiB here against 11.0 MiB at 128K with microbatch 512. Fewer tokens share each selected expert in a smaller microbatch, and the context is longer; the two effects were not separated.
- `q4_0` values were not run at 256K. At load they reserved about 1.4 GiB more than turbo3, but at 128K most of that difference did not show up in the peak.
- Single synthetic recall run: it says nothing about coding or agent quality at this length.

## Update: faster turbo3 decode

27 November 2026. A synthetic test showed one attention call with a single query over 131K cached positions taking 22.6 ms with turbo3 values against 4.9 ms with `q4_0`. `patches-experimental/turbo3-decode.patch`, applied after the three patches below, decodes turbo3 values without the table lookup (branchless selects) and gives the `q4_0`/turbo3 pair the same decode kernel tiling as `q4_0`. In the reviewer's own build the synthetic call fell to 5.7 ms (`q4_0` 4.8 ms). No extra memory is reserved. All the Metal tests listed below passed with the prefill switch at 1 and at 0. The first change keeps the same arithmetic; the new tiling changes the order of additions, so results can differ in the last bits.

| Model run | turbo3 decode before | turbo3 decode with the patch | `q4_0` decode |
| --- | ---: | ---: | ---: |
| 7,895 prompt tokens | 4.22 tok/s | 7.03 tok/s | 8.09 tok/s (same session) |
| 129,987 prompt tokens | 2.82 tok/s | **4.71 tok/s** | 5.42 tok/s (earlier run) |

Both turbo3 runs with the patch recalled 8/8 codes and wrote the same text as before, with the same tokens. At 128K the peak footprint stayed at 4,573 MiB, about 190 MiB below `q4_0`, with no swap growth, and prompt processing was 74.5 tok/s. Mean log-probability differences from the earlier turbo3 runs were 0.004 at 8K and 0.012 at 128K (largest single difference 1.02). One run per row and about 90 decoded tokens each, so the exact speeds are uncertain. The patch was written with Freebuff, see the disclosure below.

So at 128K turbo3 now decodes at about 87% of the `q4_0` speed in these runs, processes prompts at the same speed and uses about 190 MiB less.

## Update: 128K, turbo3 against `q4_0` values

27 November 2026. One run each of the same document (129,987 prompt tokens, 8 planted codes), 131,072 context, microbatch 512, K `q4_0`, banded attention in prompt processing, 3,072 MiB expert cache, 16 GB M1 Pro. The build had all the patches below, plus unpublished decode options that were left off.

| | turbo3 V | `q4_0` V |
| --- | ---: | ---: |
| Codes recalled | 8/8 | 8/8 |
| Prefill tok/s (time) | 74.4 (29 min 8 s) | 74.3 (29 min 11 s) |
| Decode tok/s, 92 tokens | 2.82 | 5.42 |
| Peak process footprint, MiB | 4,574 | 4,766 |
| Metal KV cache / compute buffer at load, MiB | 1,033 / 2,996 | 1,220 / 3,523 |
| Swap growth, sampled every 15 s | none | 86 MiB |
| Lowest free memory reported by macOS | 14% | 11% |
| SSD reads in prefill | 314 s, 1,402 GiB | 324 s, 1,388 GiB |

Both wrote the same 92 tokens (mean absolute log-probability difference 0.017). Under load turbo3 saved about 190 MiB of peak footprint, close to the difference in the KV cache; the larger difference in the compute buffer reserved at load did not show up in the peak. The footprint leaves out file-mapped weights.

- **Both completed a populated 128K prompt on 16 GB**, with little free memory left. Other large apps may not fit alongside.
- **Turbo3 decoded at about half the speed** at this length. Decode uses dense attention with both formats by default. A third run of turbo3 with an unpublished option that uses banded attention in decode too gave 2.96 tok/s (against 2.82), with the same 8/8 codes, the same text and the same peak footprint. So the attention variant does not explain the gap. The cause was later traced to decoding turbo3 values, see the update above.
- SSD reads were about 18% of prefill time.
- These are single runs of a synthetic recall test. They say nothing about coding or agent work at 128K. Prefill was faster here than at 32K below; the reason is not known.

## Update: faster turbo3 prompt processing

`patches-experimental/turbo3-prefill.patch` applies after the two patches below. In prompt processing it converts the cached keys and values to `f16` once per attention call instead of inside every small query tile. It also applies to other quantized pairs in banded attention, including `q4_0`/`q4_0`. `GGML_METAL_FA_KV_F16=0` restores the earlier paths.

On the sources with all three patches, FLASH_ATTN_EXT_BANDED 321, FLASH_ATTN_EXT 5,428, SET_ROWS 174 and TURBO_WHT 3 passed with the switch at 1 and at 0. At 31,883 tokens, on the same document as the table below, turbo3 prefill rose from **52.0 to 65.2 tok/s** (one run each; `q4_0` had measured 68.1 earlier). Peak footprint stayed at 3,617 MiB, and the 92 output tokens and log-probabilities were identical to the run before the change. At 7,895 tokens, turbo3 and `q4_0` outputs were also identical before and after. The speed figure comes from one pair of runs. The patch was written with Freebuff, see the disclosure below.

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

So banded attention removes the extra peak memory of the dense turbo3 path (about 384 MiB here), and at 32K turbo3 uses about 50 MiB less than `q4_0` values. By itself it did not make prompt processing faster: turbo3 was about a quarter slower than `q4_0` there until the prefill update above. The speeds are single runs. The Metal compute buffer reported at load was the same for the dense and banded builds, so it does not show this difference. This section stops at 32K; see the 128K update above.

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
git -C .llama.cpp-turbo3 apply "$PWD/patches-experimental/turbo3-prefill.patch"
git -C .llama.cpp-turbo3 apply "$PWD/patches-experimental/turbo3-decode.patch"
cmake -S .llama.cpp-turbo3 -B build/turbo3 -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
cmake --build build/turbo3 --target llama-server --parallel 4
```

The measurements started the server with the same flags as the `long` profile apart from `--cache-type-v turbo3` and no checkpoints. Starting it through `serve.sh` was not tested:

```bash
INKLING_SERVER=build/turbo3/bin/llama-server INKLING_CACHE_TYPE_V=turbo3 ./serve.sh long "$(./download.sh iq2)"
```
