# Writing the expert slot map without a GPU copy

Apple M1 Pro, 16 GB, 26 November 2026. Experimental: not part of `build.sh`.

## Change

[`ssd-gap.md`](ssd-gap.md) showed that splitting GPU work into many small command buffers with pauses slows it down. SSD streaming does this: for every MoE layer it reads the selected expert ids back, loads missing experts and then uploads the slot map with a small GPU copy.

`patches-experimental/ssd-direct-slot-map.patch` replaces that last upload with a `memcpy`. On Apple Silicon the destination is in memory shared by CPU and GPU. It applies to the banded build sources, runs only with `LLAMA_INKLING_SSD_DIRECT=1` and SSD streaming, and otherwise keeps the old path. The server logs `writing the slot map directly to shared memory` once when it is used.

The write follows the read-back of the ids, which waits for the queued GPU work, and the work that reads the slot map is submitted after it. A separate AI review of the code, with nothing built or run, found no race. It could not verify from this code Apple's guarantees about when CPU writes to shared memory become visible and about the completion order of command buffers.

## Summary

**No reliable speed gain.** Two first small series showed decode 12% to 30% faster, and this note first reported that. A longer series run afterwards, with steadier memory conditions, did not confirm it. Output was identical in every run.

## Longer series

About 1,900 prompt tokens, 128 generated, 4,096 context, microbatch 256, three prompts per server, eight servers in the order off, on, on, off, on, off, off, on. The order and the comparison of the four adjacent pairs were fixed before the run. Speeds are total generated tokens over total decode time for each server.

| Pair | Direct write off, tok/s | On, tok/s | Change |
| --- | ---: | ---: | ---: |
| 1 | 4.80 | 5.13 | +7.0% |
| 2 | 3.85 | 4.61 | +19.6% |
| 3 | 5.85 | 4.88 | -16.6% |
| 4 | 5.89 | 6.28 | +6.6% |
| Mean of 4 servers | 5.10 | 5.22 | +2% |

The median of the pair changes was +6.8%. The ranges overlap almost entirely (3.85 to 5.89 off, 4.61 to 6.28 on), and two servers with the same setting differed by up to about 50%. Prompt processing averaged 75.2 tok/s off and 75.4 on. Swap grew by at most 0.6 GB in any server, and lowest free memory was 22% to 32%. All eight servers produced the same tokens and log-probabilities on all three prompts (384 tokens per server). Peak footprint did not change.

A small gain cannot be ruled out, but this test cannot separate it from the variation between repeats.

## First series

Same binary for both settings, greedy, K and V `q4_0`, 3,072 MiB expert cache. On and off ran in mirrored order.

**About 1,900 prompt tokens**, 64 generated, 4,096 context, microbatch 256, three prompts per server, two servers per setting (off, on, on, off):

| Direct write | Decode tok/s, mean of 6 | Range | Mean time per token |
| --- | ---: | --- | ---: |
| off | 4.47 | 3.63 to 4.96 | 223 ms |
| on | 5.80 | 5.23 to 6.56 | 171 ms |

The ranges did not overlap. All tokens and log-probabilities were identical between on and off (64/64 on each prompt, difference 0.0), and between the two runs with it off.

The runs with it off also saw more memory pressure: lowest free memory 8% and 12%, against 15% and 31% with it on. Swap grew by about 3.4 GB in each off server and by 1.6 and 0 GB in the on servers. The cause was not isolated, so part of the difference may come from it.

**7,895 prompt tokens**, 94 generated, microbatch 512, one prompt per server (on, off, off, on):

| Direct write | Decode tok/s | Mean |
| --- | --- | ---: |
| off | 7.71, 5.80 | 6.76 |
| on | 8.22, 6.93 | 7.58 |

Here the ranges overlap. All four runs recalled 8/8 planted codes with identical tokens and log-probabilities. Peak footprint did not change.

## Reading

In the first series at about 1,900 tokens, prompt processing was also slower in the runs with the direct write off. Together with the difference in memory pressure, this suggests that machine conditions produced most of that gap. In the longer series prompt processing matched between settings and the decode gap mostly disappeared.

The patch stays here as an experiment: it does not change the output, but it is not shown to be faster. A [later trace](gpu-keepwarm.md) showed that it removes the 47 slot-map copies per token, while the 48 read-backs of expert ids and the model computation remain. Missing experts are not a GPU upload with SSD streaming: they are read straight into shared memory. An earlier version of this note said otherwise. The copies used little GPU time, and the CPU work around them did not get shorter. Not tested: the public non-banded build, other hardware, and starting through `serve.sh`.

## Trying it

These commands were not run in this exact form.

```bash
INKLING_VARIANT=banded ./build.sh
cp -R .llama.cpp-banded .llama.cpp-direct
git -C .llama.cpp-direct apply "$PWD/patches-experimental/ssd-direct-slot-map.patch"
cmake -S .llama.cpp-direct -B build/direct -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
cmake --build build/direct --target llama-server --parallel 4
```

```bash
LLAMA_INKLING_SSD_DIRECT=1 INKLING_SERVER=build/direct/bin/llama-server ./serve.sh long "$(./download.sh iq2)"
```
