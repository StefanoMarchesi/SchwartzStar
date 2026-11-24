# SchwartzStar v0.1.0-alpha

Experimental local inference for Inkling 2 Mini with SSD expert streaming on Apple Silicon. The reference machine was a 16 GB MacBook Pro with an M1 Pro. [Setup](README.md) · [Runtime measurements](results/schwartzstar.md) · [Coding probes](results/coding-agent.md)

## Reproduce the build

`build.sh` checks out llama.cpp at [ae6a7ad2c5754dd8db93d04c84fda70435864f6e](https://github.com/ggml-org/llama.cpp/tree/ae6a7ad2c5754dd8db93d04c84fda70435864f6e), applies the numbered patches in order, and builds `llama-server`. It enables Metal on Apple Silicon and builds the macOS process-footprint sampler. The Linux path builds a CPU reference. Use `INKLING_SOURCE_DIR` and `INKLING_BUILD_DIR` to select dedicated directories if needed.

```bash
./build.sh
MODEL=$(./download.sh iq3)
INKLING_CTX=8192 INKLING_BATCH=128 INKLING_CACHE_MIB=3072 ./serve.sh quality "$MODEL"
```

The download script keeps the weights outside source control, verifies the exact byte count and SHA-256, and resumes interrupted downloads. Its manifest is the source of truth for the downloadable checkpoints.

| Variant | GGUF | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| IQ1 | [Inkling-2-Mini-UD-IQ1_S](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/8078d149286eec7193795413b9e0643da6d11fa1/Inkling-2-Mini-UD-IQ1_S.gguf) | 7,764,293,056 | `17a7ff34acb0e1fe878309962458e0633172c7f93ffe3b1c4f5f6b317dc317db` |
| IQ2 | [Inkling-2-Mini-UD-IQ2_XXS](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/8078d149286eec7193795413b9e0643da6d11fa1/Inkling-2-Mini-UD-IQ2_XXS.gguf) | 8,486,237,632 | `2f953a44281cf861b19ccfdb2ce65f7648e1e92bda4af3729893604fd907c544` |
| Q2 | [Inkling-2-Mini-UD-Q2_K_XL](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/8078d149286eec7193795413b9e0643da6d11fa1/Inkling-2-Mini-UD-Q2_K_XL.gguf) | 9,138,716,608 | `b42841608ea380ddde5a3f1125081357128a3c71b80d006246de4e31292fc6c1` |
| IQ3 | [Inkling-2-Mini-UD-IQ3_XXS](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/8078d149286eec7193795413b9e0643da6d11fa1/Inkling-2-Mini-UD-IQ3_XXS.gguf) | 11,663,310,784 | `fd9ccedae17c1bb89744612fbb46730b5275f2a342133d0419776bd1ae311657` |
| IQ4 | [Inkling-2-Mini-IQ4_NL](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF/resolve/e8a2cecf0e78914f07cc27b9be711a9b2c7561d9/Inkling-2-Mini-IQ4_NL.gguf) | 17,627,940,768 | `cbdf3fa9dc50a4124977980f37da4f1fa764b0cfe800d275cd43d26ce2a81af4` |

## What was tested

- The IQ2 comparison used the same checkpoint with resident weights and SSD streaming. It produced 24 identical greedy tokens and an observed maximum top-k probability difference of zero, with 48/48 routed applications and working Completion and Responses endpoints.
- At 4,096 configured positions, a brief IQ4 chat reached its first visible text in 13.97 seconds and decoded at 2.62 tok/s, with a peak process footprint of 2,039 MiB. A separate 3,000-token IQ4 prefix took 1,120 seconds on the earlier u4 profile.
- The current cache sizing and large-microbatch path made IQ3 and IQ2 prefill substantially faster on the 512-token probe. The report records effective prompt length, throughput, output, RSS, footprint, and system swap. [Measurements](results/schwartzstar.md)
- IQ4 streamed with 131,072 positions allocated and 32 evaluated tokens at 3,175 MiB peak process footprint using Q4_0 keys, F16 values, and a 256 MiB expert cache. Populated 128K remains a separate validation target. [Long context](LONG-CONTEXT.md)
- The coding probes report native tool calls and an independent structured-edit benchmark. A rising token rate does not establish patch quality. [Coding results](results/coding-agent.md)

## Credits and licenses

This repository includes source changes and utility scripts under the MIT license in [LICENSE](LICENSE), with llama.cpp’s existing copyright notice preserved. The implementation builds on llama.cpp’s [Inkling 2 Mini support](https://github.com/ggml-org/llama.cpp/pull/32694) and `adishik`’s [shared expert cache](https://github.com/ggml-org/llama.cpp/issues/32744). The direct SSD path takes inspiration from [DwarfStar by Salvatore Sanfilippo](https://github.com/antirez/ds4/blob/main/docs/SSD_STREAMING.md) and [Alex Schiltmans’s Metal expert streaming](https://github.com/alexschiltmans/llama.cpp-per-layer-experts/blob/per-layer-experts/docs/moe-ssd-streaming.md).

Model weights are distributed separately. [Thinking Machines Lab’s base model](https://huggingface.co/thinkingmachines/Inkling-2-Mini) lists Apache-2.0 and links its [Model Acceptable Use Policy](https://thinkingmachines.ai/model-acceptable-use-policy/). [Unsloth’s GGUF card](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) also lists Apache-2.0. The [ggml-org IQ4_NL card](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF/tree/e8a2cecf0e78914f07cc27b9be711a9b2c7561d9) links the base model. Download the checkpoint from its publisher and check the model’s terms for your use.
