# Inkling 2 Mini at 128K–1M on M1 Pro

Objectives to verify on 24 November 2026: local generation on an M1 Pro 16 GB with the usual applications open, a physical footprint of approximately 4–5 GB, preference for the 4-bit GGUF, 524,288 context tokens as the first target, and possibly 1,048,576. Speed must be measured with the context actually populated.

## Sizing

The Mini uses 48 layer applications: 40 with a 1,024-token window and 8 global applications. There are 8 KV heads with 128 elements each. For each token, global KV requires `8 × 2 × 8 × 128 × byte_per_element`. Recurrent-convolution memory is separate for each application. This remains true even when expert weights are read exclusively from SSD. [vLLM implementation and architecture](https://recipes.vllm.ai/thinkingmachines/Inkling-2-Mini)

| KV format for K and V | Global KV at 512K | Global KV at 1M | Evidence |
| --- | ---: | ---: | --- |
| F16 | 16.00 GiB | 32.00 GiB | Prototype format |
| Q8_0 | 8.50 GiB | 17.00 GiB | Calculated from 34 bytes per 32 values; requires a compatible attention path |
| Q4_0 | 4.50 GiB | 9.00 GiB | Calculated from 18 bytes per 32 values; requires a compatible attention path |
| Turbo3 | ~3.13 GiB | ~6.25 GiB | Extrapolated from ~800 MiB of global KV at 128K, reported by a PR on the Atomic fork |
| Ideal 2-bit | 2.00 GiB | 4.00 GiB | Theoretical limit before scales, metadata, and alignment |

The table excludes local KV, convolution, scratch space, models, allocators, and drivers. Keeping local SWA KV in FP16 costs approximately 160 MiB of payload. Even Turbo3 estimated at 512K, added to the 1.69 GiB of non-routed weights in the IQ4_NL GGUF, exceeds 4.8 GiB before any expert cache. The same calculation at 1M puts even Turbo3 KV alone over budget. [Atomic Turbo3 PR](https://github.com/AtomicBot-ai/atomic-engine/pull/101)

```bash
python3 budget.py 524288 1048576 --resident-kv-mib 1024 --ssd-gib-s 3
```

With 1 GiB of resident global KV and a hypothetical SSD sustaining 3 GiB/s, a dense scan with Q4_0 K/V has an I/O limit of approximately **0.86 tok/s at 512K** and **0.38 tok/s at 1M**. Turbo3 would provide approximately **1.41** and **0.57 tok/s**. These are optimistic limits before expert reads, computation, latency, and prefill. At 5 tok/s with the same throughput, the complete SSD path would have only about 0.6 GiB available per token. Reaching that rate requires avoiding a full reread on every token.

## Verified 128K allocation-only probe

The current verified result is an allocation-and-short-inference probe, not a 128K populated-context benchmark. With IQ4_NL, Q4_0 keys and F16 values, microbatch 1, and a 256 MiB expert cache, the server allocated **131,072** positions and evaluated **32** input tokens. It emitted eight tokens, with first text in **14.76 s**, prefill **2.17 tok/s**, decode **2.28 tok/s**, and all **48/48** routed applications observed. Peak **physical footprint** was **3,175 MiB**, peak **RSS** **4,907 MiB**, and swap changed from 2,292.1 to 2,284.1 MiB. Metal KV buffers allocated 2,624 MiB for the eight global applications and 128.12 MiB for the local window. See the [source measurement](results/schwartzstar.md).

The field names are intentional: “allocated” refers to the context capacity, “evaluated” to the input actually processed, “physical footprint” to the process high-water mark, and “RSS” to resident set size. Retrieval and inference with 128K populated remain open. The current attention graph builds a relative-bias tensor whose size grows with both microbatch and populated keys. The upstream experimental Metal banded-attention proposal targets that problem. [Metal banded Flash Attention proposal](https://github.com/unslothai/llama.cpp/pull/431)

```bash
IQ4=$(./download.sh iq4)
INKLING_CTX=131072 INKLING_BATCH=1 INKLING_CACHE_MIB=256 INKLING_CACHE_TYPE_K=q4_0 \
  python3 check.py run quality "$IQ4" iq4-q4k-128k-start --prefix-only 32 --tokens 8 --timeout 90
```

## How the weight quantization changes the budget

Tensor bytes are taken from the GGUF headers of the pinned revisions. The I/O column is the payload if all 8 experts for each of the 48 applications miss. A cache shared across passes can reduce it. Top-1 is Unsloth's published agreement with its UD-Q8_K_XL on the corpus used by Unsloth. It does not measure accuracy at long contexts. [Unsloth methodology and results](https://unsloth.ai/docs/models/inkling-2-mini#quantization-analysis)

| GGUF | File, GB | Weights outside experts, GiB | I/O with all misses, MiB/token | Unsloth top-1 |
| --- | ---: | ---: | ---: | ---: |
| ggml-org IQ4_NL | 17.63 | 1.69 | 1,762 | — |
| Unsloth UD-Q4_K_XL | 17.63 | 1.69 | 1,762 | 98.0% |
| Unsloth UD-IQ3_XXS | 11.66 | 1.26 | 1,239 | 94.7% |
| Unsloth UD-Q2_K_XL | 9.14 | 1.15 | 974 | 91.6% |
| Unsloth UD-IQ2_XXS | 8.49 | 1.14 | 878 | 90.6% |
| Unsloth UD-IQ1_S | 7.76 | 1.14 | 795 | 89.4% |

With the expert cache fixed, moving to IQ3 saves approximately 443 MiB of resident weights and 30% of expert payload in the all-misses case. Q2_K_XL saves approximately 553 MiB and 45% versus IQ4_NL. Actual tok/s also depends on Metal kernel performance for each quantization and the hit rate. To assess 4/3/2-bit quality without conflating quantization and length, use the same prompt set, a short-context reference, and a retrieval-and-synthesis test with distant positions at 128K/512K.

## Priority experiments

| Technique | What it solves | Status for Inkling 2 Mini on M1 Pro |
| --- | --- | --- |
| Turbo3 compressed KV + attention kernel reading quantized blocks | Reduces KV RAM and I/O without removing positions | Open Atomic PR. The author reports 128K and ~11.7 tok/s, on hardware and under conditions not given in the PR summary. No measurement here at 512K or combined with expert streaming. [PR](https://github.com/AtomicBot-ai/atomic-engine/pull/101) |
| Relative block bias or banded attention | Reduces temporary buffers. Allows Flash Attention while retaining bias and sinks | Experimental Unsloth patches and upstream proposals exist. Conditions change with microbatch, wrapped SWA cache, and KV quantization. Verify Metal backend and numerical parity before combining them. [Unsloth #385](https://github.com/unslothai/llama.cpp/pull/385) · [Unsloth #386](https://github.com/unslothai/llama.cpp/pull/386) · [Upstream #32752](https://github.com/ggml-org/llama.cpp/pull/32752) |
| KV on SSD with double buffering and online softmax | Keeps the full logical context while limiting RAM. Every decode token still reads cold global KV | Implemented for Inkling 2 Small on DGX Spark with NVFP4 KV, a resident threshold, and double buffering. It is a reference for a Metal port. The Mini requires an application-ID layout and kernels compatible with its controllers. [Fork code and measurements](https://github.com/Baekpica/ds4-dfm-rs/blob/main/docs/inkling2-vq.md) |
| Asymmetric 2-bit KV, KIVI-style | Uses different quantization axes for keys and values. May reduce cold KV further | Research on other models. Requires a kernel that avoids dequantizing the full sequence and a long-distance retrieval test. [KIVI](https://proceedings.mlr.press/v235/liu24bz.html) |
| Query-driven page selection, Quest or InfiniGen style | Reduces KV read per token. It is the most promising lever for several tok/s after 512K | Approximation. For Inkling, selection must include relative-bias contribution, keep the sink once, and cover remote positions well. Requires a quality measurement against dense attention. [Quest](https://proceedings.mlr.press/v235/tang24l.html) · [InfiniGen](https://www.usenix.org/conference/osdi24/presentation/lee) |
| DSpark and prefix reuse | DSpark can amortize scanning across several verified tokens. Prefix reuse reduces repeated prefill | The drafter uses internal loop taps and additional memory. The gain depends on acceptance rate. Saving prefixes to disk does not reduce KV required by one active sequence. [DSpark architecture](https://vllm-project.github.io/2026/11/20/inkling2-mini.html) · [Unsloth](https://unsloth.ai/docs/models/inkling-2-mini#inference-speed) |
| vLLM Metal with TurboQuant | Provides quantized-KV kernels on Apple Silicon | The public matrix does not include Inkling 2 Mini. A codec for other models does not imply compatibility with its loops and relative bias. [Matrix](https://github.com/vllm-project/vllm-metal/blob/main/docs/supported_models.md) · [TurboQuant](https://docs.vllm.ai/projects/vllm-metal/en/latest/turboquant/) |

## KV-port design

1. Keep SWA ring buffers, convolution states, and a configurable global-KV prefix/recent window in memory. Index SSD by `(session, logical_application, page)`, never by physical layer alone.
2. Use the Turbo3 branch as the reference for compressed format and Metal computation. Start at 2K and 32K with one sequence, then compare at 128K. Separate the effect of KV quantization from paging by using the same format on both paths.
3. Implement the dense page path with two shared I/O buffers, F32 softmax state, and bias/sink in every head. The attached Python reference checks exact fusion with synthetic positional bias and a sink over 4,099 rows. It does not run the model.
4. During decode, measure bytes/token, expert hit rate, and SSD time. If dense scanning is the bottleneck, add page selection with a dense reference path. During prefill, use query and KV blocks to reuse each read, keeping this separate from expert microbatching.
5. Validate genuinely populated contexts at 128K, 256K, 512K, and 1M. Record time to first token, prefill and decode tok/s, physical footprint, swap, bytes read, and multi-hop retrieval tests at distant positions. With DSpark, also measure accepted-draft share and the cost of rolling back every cache per pass.

The package runs the expert-streaming path with a short context and estimates the long-path budget. The Turbo3/SSD runtime port and M1 Pro long-context numbers remain open.
