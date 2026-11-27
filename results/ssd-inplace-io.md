# Reading router ids and writing the slot map in place

Apple M1 Pro, 16 GB, 27 November 2026. Experimental, off by default, not part of `build.sh`.

## Prior work

This is not new. `LLAMA_MOE_CACHE_DIRECT_IDS` in [Alex Schiltmans's per-layer expert streaming](https://github.com/alexschiltmans/llama.cpp-per-layer-experts/blob/per-layer-experts/docs/moe-ssd-streaming.md) reads the router ids from shared memory and writes the slot map in place, and also drops the second wait. The patch here keeps a synchronize call before the read.

## Change

`patches-experimental/ssd-inplace-io.patch` applies to the banded build sources. With `LLAMA_MOE_CACHE_INPLACE_IO=1`, SSD streaming reads the router ids directly from the shared Metal buffer instead of queuing a GPU copy, and writes the slot map directly instead of uploading it. It falls back to the old path when a buffer is not visible to the CPU. In an [earlier trace](gpu-keepwarm.md) these copies were about 95 of 192 command buffers per decoded token. An [earlier patch](direct-slot-map.md) removed only the slot-map upload and showed no reliable gain.

## Speed

Eight servers in the order off, on, on, off, on, off, off, on. About 1,900 prompt tokens, one prompt, 512 generated tokens, 4,096 context, microbatch 256, K and V `q4_0`, 3,072 MiB expert cache. The [GPU keep-alive](gpu-keepwarm.md) was on in all eight, so this is a gain on top of it. The order, the comparison of adjacent pairs and the rule for claiming a gain (all four pairs faster and identical output) were fixed before the run. The binary was a build with the other experimental patches and unpublished options, left off, plus two small unpublished code cleanups in the expert cache.

| Pair | Decode off, tok/s | Decode on, tok/s | Change | Prefill off / on, tok/s |
| --- | ---: | ---: | ---: | --- |
| 1 | 7.09 | 9.10 | +28% | 75.5 / 79.3 |
| 2 | 5.45 | 8.60 | +58% | 56.5 / 80.3 |
| 3 | 4.84 | 7.21 | +49% | 73.7 / 65.0 |
| 4 | 5.44 | 5.50 | +1% | 57.6 / 69.5 |
| Mean of 4 servers | 5.70 | 7.61 | | 65.8 / 73.5 |

- Decode was faster with the option in all four pairs, with a median of +39%. **The size of the gain is uncertain:** it ranged from 1% to 58%.
- The machine was in use, and speeds varied a lot between runs with the same setting (4.8 to 7.1 tok/s off, 5.5 to 9.1 on). Prompt processing also moved from run to run, which suggests part of the spread comes from the machine.
- All eight runs produced the same 512 tokens and log-probabilities.
- Across all servers the ranges overlap: the slowest run with the option (5.50) was below the fastest without it (7.09). Only the adjacent pairs all favour it.

## Correctness

A build of the public banded sources with only the keep-alive patch (off) and this patch gave, at 7,895 prompt tokens, the same 94 tokens and log-probabilities with the option off and on, and the same as an earlier run saved before this patch, with 8/8 planted codes. One document at one length. SSD streaming with long contexts and other expert cache sizes was not tested with this option.

The CPU read relies on the scheduler and the synchronize call having waited for the GPU work that wrote the ids. This was checked by reading the code and by the identical outputs above, not by a stress test.

## Trying it

These commands were not run in this exact form.

```bash
INKLING_VARIANT=banded ./build.sh
cp -R .llama.cpp-banded .llama.cpp-inplace
git -C .llama.cpp-inplace apply "$PWD/patches-experimental/ssd-inplace-io.patch"
cmake -S .llama.cpp-inplace -B build/inplace -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
cmake --build build/inplace --target llama-server --parallel 4
```

```bash
LLAMA_MOE_CACHE_INPLACE_IO=1 INKLING_SERVER=build/inplace/bin/llama-server ./serve.sh long "$(./download.sh iq2)"
```

The patch was written with [Freebuff](https://freebuff.com/?ref=ref-b181061b-0b36-403c-9e01-a696e0b1b6fc&via=rewards_link) running DeepSeek V4.1 Flash, then reviewed and measured with another AI assistant. Disclosure: I am not affiliated with Freebuff. That is my invite link, and Freebuff's programme may reward me if you use it; as far as I know this is only its points collection for merchandise, and the app also offers usage hours for invitations.
