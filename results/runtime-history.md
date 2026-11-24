# Reproducible summary — phases 0 and 1

Test date: 24 November 2026. Machine: MacBookPro18,1, Apple M1 Pro, 16 GB, macOS 27.3, arm64, Xcode 27.2.

## Location

| Item | Path |
| --- | --- |
| Project and repository | Checkout root |
| GGUF (local model link) | `./models` |
| llama.cpp sources | `./llama.cpp` |
| Build | `./build` |

`models/` contains only a symbolic link to the GGUF and is excluded from the repository. The USB benchmark is postponed: the measurements in this phase use the configured local model storage.

## Verified model

`Inkling-2-Mini-UD-IQ2_XXS.gguf`, 8.486.237.632 bytes, SHA-256 `2f953a44281cf861b19ccfdb2ce65f7648e1e92bda4af3729893604fd907c544`, verified after copying to the internal SSD.

## Commands

```bash
export INKLING_SOURCE_DIR="./llama.cpp"
export INKLING_BUILD_DIR="./build"
JOBS=4 ./build.sh 2>&1 | tee results/build.log
python3 kv_reference.py
python3 budget.py
```

`build.sh` clones `llama.cpp` at commit `ae6a7ad2c5754dd8db93d04c84fda70435864f6e`, applies `patches/*.patch` in a temporary index, and builds `llama-server` plus `inkling-footprint` with Metal enabled. The patches do not touch the checkout used by the repository.

## Static verification of router, synchronization, and Metal slot ordering

In `llama_moe_cache::impl::copy()` the selected-expert readback is synchronous with respect to the device:

```cpp
ids.resize(ggml_nelements(sel));
ggml_backend_tensor_get_async(backend, sel, ids.data(), 0, ggml_nbytes(sel));
ggml_backend_synchronize(backend);
```

`ggml_backend_synchronize()` follows the readback immediately and precedes both `plan()` and writes to `bank->data`. Each `pread` therefore writes to slots that no Metal kernel is still reading: the race hypothesis is not confirmed and no additional synchronization was added. The synchronization cost is already paid once per application layer because it is required for the router readback.

Two properties remain to be observed at runtime and cannot be added statically: a plan's slots are disjoint (guaranteed by `plan()`, which assigns `victim()` and free slots), and the `bank` buffer is shared between host and GPU (enforced by the patch with `--no-op-offload` on a single device).

## Application count

`layers.resize(n_layer_all)` brings `model.layers` to 48 entries: 24 physical expert layers, of which the 12 core layers run three times, for 36 applications, plus 12 non-repeated layers. The expected value is therefore **48/48 applications**, with 48 distinct expert tensors (gate_up and down for 24 physical layers). Before patch 0003 the log counted layers with at least one miss, not applications executed: a layer entirely in cache disappeared from the count. Patch 0003 counts lookups per application and prints the per-layer histogram, so a missing pass becomes visible and an eventual 24/24 must be investigated, not accepted.

## Test instrumentation

`check.py` now measures:

- TTFT from the first content token in a `stream: true` request, with prefill and decode from the server `timings`;
- three consecutive greedy requests in the same process, with text comparison across repetitions;
- both endpoints, `/completion` and `/v1/responses`;
- a fixed-token-prefix probe, built with `/tokenize` and `/detokenize`, to separate prefill;
- a probe beyond the SWA window (`--swa-tokens`);
- separate I/O counters for prefill and decode (patch 0003);
- swap, memory pressure, and load average before and after the test, plus peak RSS and physical footprint during it.

`check.py compare` walks every position and reports every divergence (token, top-k, delta) instead of stopping at the first one.

## Defects fixed during validation

1. `serve.sh`: `"${extra[@]}"` with an empty array under `set -u` interrupted the resident profile on macOS bash 3.2. It now uses `${extra[@]+"${extra[@]}"}`.
2. `patches/0004-metal-shared-buffers.patch`: the cache rejected every streaming startup with “requires a single GPU with shared buffers”, because on Metal all buffer types report `is_host = false`. The check now accepts the Metal buffer type (actual name `MTL0`), which on Apple Silicon is unified memory, and records it in the log. Without this correction the `minimum`/`quality` profile could not start on this Mac at all.
3. `patches/0003-io-counters.patch`: separate I/O counters for prefill and decode, plus a lookup histogram per application.

## Preliminary results (machine under load)

The tests ran with load 19–28, system swap between 9.8 and 14.3 GB, and an external process at roughly 260% CPU. The timings are preliminary and should be repeated with a lower load.

Gate: same IQ2_XXS file, `INKLING_CTX=1024`, microbatch 1, 24 tokens, three consecutive greedy requests.

| | resident | streaming SSD (512 MiB cache) |
| --- | ---: | ---: |
| TTFT request 1 / 2 / 3 | 2,223 / 1,561 / 1,774 s | 6,111 / 6,445 / 6,180 s |
| prefill (17 tokens) | 1.856 ms (9,2 tok/s) | 6.267 ms (2,7 tok/s) |
| decode | 9,80 tok/s | 2,62 tok/s |
| peak RSS | 2.484 MiB | 1.981 MiB |
| peak physical footprint | 391 MiB | 875 MiB |
| system swap, change | +5.453 MiB | 0 MiB |
| router applications | — | 48/48, 199 lookups per application |
| reads | — | 78.798 of 89.580 MiB, 48,4% hit rate |

Greedy parity: 24 identical tokens, maximum top-k delta **0** (bit-for-bit). Both endpoints (`/completion` and `/v1/responses`) responded in both profiles.

Probes with `INKLING_CTX=2048` (fixed-token prefix 512, then 1.280 tokens, beyond the 1.024-token SWA window):

| Prefix | resident, TTFT | resident, decode | SSD, TTFT | SSD, decode |
| --- | ---: | ---: | ---: | ---: |
| 512 tokens | 14,06 s | 11,5 tok/s | 80,9 s | 3,06 tok/s |
| 1.280 tokens | 35,50 s | 14,5 tok/s | 195,1 s | 3,31 tok/s |

Both probes produced the same text as the resident profile, so parity also holds beyond the SWA window. With microbatch 4 the phase counters genuinely separate: 430.699 MiB in prefill versus 8.431 MiB in decode, 60,3% hit rate with a 1.536 MiB cache.

## Reading the numbers

- The resident profile's physical footprint (391 MiB) is not the weight cost: the model's 7.847 MiB are clean mmap pages, which macOS does not count as dirty memory. The resident profile remains outside budget because those 7.8 GB must stay in the page cache to avoid being reread from disk.
- With microbatch 1, prefill advances one token per ubatch, so the counters classify it as decode: this is why the gate shows `prefill reads = 0`. Phase separation requires a larger microbatch, as in the probes.
- With a 512 MiB cache each decode token reads ~800 MiB (48% hit rate); with a 1.536 MiB cache prefill reads ~240 MiB per token. The bottleneck identified by the handoff remains expert I/O, not compute.

## Buffer verification, not only backend verification

`MTL0` is the buffer type name, not a guarantee about an individual allocation. `patches/0005-shared-buffer-check.patch` adds `ggml_backend_metal_buffer_is_shared()`, which queries the actual Metal buffer (`ggml_metal_buffer_is_shared()` in its context), and the cache uses it on newly allocated banks: if a bank is not in Shared memory, startup stops. The log reports how many banks were checked and their buffer type.

Synchronization remains the existing sequence: `ggml_backend_tensor_get_async()` on the router followed by `ggml_backend_synchronize()`, before `plan()` and every write to the banks. It covers the Metal queue reading those slots and was not duplicated.

## Explicit request phase

`patches/0006-explicit-io-phase.patch` adds `llama_set_io_phase()`; the server sets it for every batch using the `is_prompt` flag already tracked per token. The cache no longer infers the phase from ubatch size: with microbatch 1, prefill is counted as prefill.

The log reports tokens, misses, reads, and bytes for each phase. The tokens are **evaluations per application**, raw: `prefill_app_tokens` and `decode_app_tokens`. Dividing by the 48 layers gives the tokens actually evaluated (90 and 77 in the matrix case) only if coverage is uniform, which the `per_layer` histogram verifies. The division is not rounded, so incomplete coverage remains visible as a fractional value. Tokens actually emitted are reported separately, from the server counters (`server_tokens`), not derived from the cache.

## IQ4_NL: smoke test

`INKLING_CTX=1024`, microbatch 1, 768 MiB cache, 16 tokens: 48/48 applications, 48 distinct expert tensors, 45.102 reads for 105.079 MiB, physical footprint 1.127 MiB, TTFT 8,73 s, decode 1,97 tok/s. Both endpoints responded.

## IQ4_NL: cache matrix

Same GGUF, same prompt, `ctx 1024`, microbatch 1, 16 tokens, three consecutive greedy requests. The 4.500 MiB threshold was respected by every configuration.

| cache | footprint | prefill MiB | decode MiB | tok/s | TTFT (3 reps.) | swap Δ | compressed before → after |
| ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| 256 MiB | 615 MiB | 145.088 | 124.883 | 1,46 | 11,10 / 11,46 / 9,80 s | +1.237 MiB | 6.478 → 6.140 MiB |
| 512 MiB | 875 MiB | 116.763 | 99.733 | 1,78 | 9,08 / 11,01 / 8,99 s | +136 MiB | 6.111 → 5.034 MiB |
| 768 MiB | 1.131 MiB | 90.432 | 78.274 | 1,91 | 8,96 / 8,76 / 8,76 s | +223 MiB | 4.894 → 5.493 MiB |
| 1.024 MiB | 1.391 MiB | 78.295 | 67.872 | 1,83 | 8,30 / 8,81 / 9,28 s | +89 MiB | 5.374 → 5.824 MiB |

All four configurations produced identical greedy text with top-k delta 0: the cache changes only I/O.

From 768 to 1.024 MiB, decode reads fall from 78.274 to 67.872 MiB, −13,3%, and prefill from 90.432 to 78.295 MiB, in exchange for 260 MiB of footprint. Measured decode rises from 1,91 to 1,83 tok/s, too small a gap to declare a winner from these measurements alone. **768 MiB remains the provisional default; 1.024 MiB is the candidate to compare in the next clean window**, where the I/O difference can be translated into a credible tok/s difference.

Swap is not a clean measure under these conditions: the system was already at 10–11 GB with 5–6 GB in the compressor, so changes of a few hundred MiB say little. Repeat on an idle machine, together with timings.

## IQ4_NL: 4K context with a real prompt

Allocated context **4.096**, effective prompt **~3.000 tokens** (fixed prefix built with `/tokenize` and `/detokenize`), cache **1.536 MiB**, microbatch 4.

| item | value |
| --- | --- |
| physical footprint | 2.199 MiB |
| peak RSS | 3.693 MiB |
| prefill evaluated | 3.056 tokens (146.688 evaluations per application) |
| decode evaluated | 38 tokens (1.824 evaluations) |
| prefill I/O | 2.419.116 MiB, ~792 MiB per token |
| decode I/O | 28.565 MiB, ~752 MiB per token |
| probe prefill | 1.120 s for 3.000 tokens (2,68 tok/s) |
| probe TTFT | 1.120 s |
| probe decode | 1,61 tok/s |
| router | 48/48, 38.544 lookups |
| system swap | 10.552 → 12.212 MiB |

It remains within the threshold with 2.301 MiB of headroom. Prefill dominates: at 2,7 tok/s an 8K prompt takes ~50 minutes, 16K ~1.7 hours, and 32K ~3.4 hours of read time alone. F16 KV grows only modestly (8 global layers plus 40 local layers limited to 1.024 positions): at 32K it adds about 1 GiB to the footprint, so the budget remains respected and the limit is time, not memory.

## Prefill: where the time goes and how much reuse matters

One request per test, new server, port confirmed free before each startup, identical prompt built with `/tokenize` and `/detokenize`.

At a 768 MiB cache and microbatch 1, with prefixes 512 and 1.024:

| prefix | prefill | tok/s | reads | sync | MiB per token |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 316 s | 1,62 | 169,7 s (53,8%) | 19,6 s (6,2%) | 1.137 |
| 1.024 | 693 s | 1,48 | 356,9 s (51,5%) | 57,6 s (8,3%) | 1.151 |

Reads are slightly more than half of prefill, at an effective **3,3–3,4 GB/s** on the internal SSD. The rest is Metal compute and waiting for router readback. Bytes per token do not depend on prefix length, so there is no reuse that grows with context at microbatch 1.

### Comparing microbatches at a fixed cache (1.536 MiB, prompt 512)

| test | microbatch | MiB per token | misses per app-token | reads | sync | footprint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| direct | 1 | 832 | 3,80 | 291 s | 119 s | 2.031 MiB |
| direct | 4 | 761 | 3,46 | 180 s | 19 s | 2.036 MiB |
| reverse | 4 | 761 | 3,46 | 194 s | 23 s | 2.040 MiB |
| reverse | 1 | 832 | 3,80 | 145 s | 19 s | 2.039 MiB |

Bytes and misses are **byte-identical between the two repetitions** in reverse order: they are deterministic. Timings are not (743 versus 282 s for the same configuration, with load average between 100 and 146), so they cannot be used in this window.

Microbatch 4 reduces both bytes per token (−8,5%) and misses (−9,0%) at a fixed cache: **it reuses more experts**. The reuse does not come from missing logic, because `plan()` already deduplicates experts selected by one ubatch (the `uniq` set per logical application). The gain is modest because nearby tokens select mostly disjoint experts; what really falls is synchronization cost, from 119 to 19 s, because there are 128 readbacks instead of 512.

With a 2.048 MiB cache, microbatch 8 falls to **597 MiB per token and 2,70 misses**, versus 638 MiB and 2,91 for microbatch 4 at the same cache. Ubatch 8 does not fit in 1.536 MiB: it stops at layer 20.

### Confirmation at prompt 1.024 (1.536 MiB cache)

| test | microbatch | MiB per token | misses per app-token | reads | sync | prefill | TTFT | footprint | decode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| u1 | 1 | 852 | 3,90 | 244 s | 22 s | 474 s | 474 s | 2.119 MiB | 2,10 tok/s |
| u4 | 4 | 789 | 3,59 | 211 s | 6 s | 323 s | 323 s | 2.118 MiB | 1,98 tok/s |

At the same footprint, microbatch 4 removes 7,4% of bytes, 7,9% of misses, and 32% of TTFT. Synchronization falls from 22 to 6 s. Measured decode is slightly worse (1,98 versus 2,10 tok/s), but over eight tokens and with variable load this difference is not interpretable.

### Parity with a larger microbatch

Comparison of ubatch 4 versus ubatch 1 at a 1.024 MiB cache: 16 identical greedy tokens, **maximum top-k delta 3,3·10⁻⁵**, below the 0,005 threshold but nonzero because `mul_mat` order changes. All 48 applications remain uniformly covered and the three passes preserve separate slots and states.

## Reconciliation of 430.699 MiB

That number came from the old classification by ubatch size and covered five requests (17 + 17 + ~22 prompt tokens, plus the 512 and 1.280 probes). With ~1.850 prompt tokens it is about 233 MiB per token, consistent with IQ2 and a 1.536 MiB cache. **The value was not used to choose the cache**: the matrix uses explicit counters, where phase tokens are exact.

## Chat test after rebooting the Mac

IQ4_NL profile with context 4.096, microbatch 4, 1.536 MiB cache, and four threads. The first streaming test returned 85 reasoning tokens and no visible text. Setting `reasoning_effort=none` changed the template instruction to `Thinking effort level: 0`, but the model still generated only reasoning. With a zero budget the sampler closed reasoning blocks on opening and chat began showing text. The first process confirmed **48/48 applications** in its output log.

On the next server, started with the updated `serve.sh` settings, the short Italian prompt produced a complete sentence in streaming from `/v1/chat/completions`, with first text at **13,97 s**, decode **2,62 tok/s**, and natural completion (`stop`). `/v1/responses` returned `The capital of Italy is Rome.` as visible text in **9,12 s** with a 16-token limit. The process peak was **2.039,3 MiB** and system swap at the end of the test was **8,31 MiB**.

```bash
INKLING_BUILD_DIR="./build" INKLING_PORT=18080 \
  INKLING_CTX=4096 INKLING_BATCH=4 INKLING_CACHE_MIB=1536 \
  ./serve.sh quality "./models/Inkling-2-Mini-IQ4_NL.gguf"
```

The 24 November 2026 raw logs are retained outside the repository under the run prefixes `iq4-reboot-live-20261124-214610` and `iq4-reboot-ready-20261124-215022`.

## Status

Build completed with Metal and with `LLAMA_BUILD_APP=OFF`, `LLAMA_USE_PREBUILT_UI=OFF`. Phase 1 gate passed on content (48/48, parity 0, both endpoints). With IQ4_NL, the 1K profile with a 768 MiB cache uses 1.131 MiB of footprint; timings and swap remain preliminary until the machine is idle.
