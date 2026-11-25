# Experimental Metal attention and SSD streaming pilot

An isolated Metal build combining the experimental banded-attention work with an NVFP4 compatibility patch started Inkling 2 Mini IQ2 with SSD expert streaming. This is an early runtime experiment, not a release or a coding benchmark score. The candidate is not included in the public build. The source snapshot is based on `0fbd675d6798174714654ae04b7819c80dc6f8eb`, with the existing SSD changes and a separate NVFP4 patch whose SHA-256 is `52c96a21daf3759eb177d42c13d404de77cd931b4523df5099d01468357208ec`. The banded work originates in [Unsloth llama.cpp PR 431](https://github.com/unslothai/llama.cpp/pull/431).

The machine was an Apple M1 Pro with 16 GB unified memory. Each comparison ran one model at a time, using the same IQ2 GGUF, prompt hash, greedy sampling, SSD expert-cache size of 3,072 MiB, microbatch of 512, Q4_0 keys and F16 values. The baseline used the current SSD build with flash attention off; the candidate used the isolated build with flash attention on. Outputs matched in each comparison, but the sample is too small to establish numerical equivalence or coding quality.

| Prompt / allocated context | Run order | Build | Prefill | First text | Decode | Peak RSS |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 2,048 / 8,192 tokens | First | Candidate | 112.24 tok/s | 18.25 s | 5.57 tok/s | 4,626 MiB |
| 2,048 / 8,192 tokens | Second | Baseline | 88.15 tok/s | 23.25 s | 4.66 tok/s | 4,635 MiB |
| 2,048 / 8,192 tokens | First | Baseline, repeat | 87.56 tok/s | 23.40 s | 4.59 tok/s | 4,688 MiB |
| 2,048 / 8,192 tokens | Second | Candidate, repeat | 109.19 tok/s | 18.77 s | 6.31 tok/s | 4,703 MiB |
| 8,192 / 16,384 tokens | First | Baseline | 79.49 tok/s | 103.08 s | 4.83 tok/s | 4,670 MiB |
| 8,192 / 16,384 tokens | Second | Candidate | 98.53 tok/s | 83.17 s | 6.19 tok/s | 4,891 MiB |

The two 2K pairs have opposite run order; the 8K result is one pair. System cache and memory pressure varied, and these short synthetic continuation prompts do not establish performance on a long agent conversation. All runs routed all 48 layer applications through the SSD path. The candidate and baseline differ in source version and attention implementation, so the measured difference cannot be attributed to a single change.

The candidate passed 73 NVFP4 `MUL_MAT_ID` cases and 10 short banded-attention cases against the CPU backend, including attention sinks and several KV types. In a separate instrumented short inference, the scheduler assigned banded attention to Metal during prefill and dense attention to Metal during decode. This confirms selection in that example, not the absence of fallbacks for other positions, cache types or context lengths.

At 131,072 allocated positions, Q4_0 keys, F16 values, microbatch 64 and a 2,304 MiB expert cache, another 2,048-token continuation produced matching output: 32.06 tok/s prefill for the candidate and 29.14 tok/s for the baseline in one pair. Their peak physical footprints were 5,428 and 5,439 MiB respectively. The candidate run completed inference but exceeded the check script's default 4,500 MiB footprint target. Allocating 128K positions is not the same as evaluating 128K input tokens.

The candidate also started with 262,144 positions allocated, Q4_0 keys, F16 values, microbatch 32 and a 2,304 MiB expert cache. A short smoke processed 128 input tokens and generated eight, with all 48 routed applications observed. Peak RSS was 8,677 MiB and peak physical footprint 7,869 MiB. System-wide swap increased by about 942 MiB during the run; this system measurement cannot be attributed solely to the model process. This establishes neither successful processing of a 200K-token input nor acceptable long-context quality. An earlier attempt with a 1,536 MiB expert cache stopped at startup because this configuration requires at least 2,179 MiB.

With 131,072 positions allocated, the candidate also passed the small [mini-SWE-agent shell fixture](mini-swe-agent-smoke.md) in 64.70 seconds with three tool calls. This checks the read, edit and test cycle, not realistic coding quality or a populated 128K context. The earlier baseline smoke used a different context and microbatch, so its timing is not directly comparable.

With 204,800 positions allocated and microbatch 64, the candidate processed the same 2,048-token continuation at 29.34 tok/s and produced the same output as the 131,072-position candidate run. Peak RSS was 7,753 MiB, peak physical footprint 6,904 MiB and the peak increase in system-wide swap was 47.5 MiB. A separate mini-SWE-agent shell smoke at this allocation passed in 69.56 seconds with three tool calls; peak RSS was 7,704 MiB and system-wide swap rose by up to 257 MiB. These single runs started under different system conditions and used short contexts: they do not demonstrate a populated 200K context or establish real-world coding quality.

For the same 2,048-token prompt and 204,800-position allocation, candidate microbatch 128 reached 43.18 tok/s; microbatch 256 reached 65.20 and 65.93 tok/s; microbatch 512 reached 62.72 and 72.05 tok/s. The 256 and 512 repeats reversed their order. All runs had the same prompt hash and output. Peak RSS was about 7,881 MiB for microbatch 256 and 7,897–7,901 MiB for 512. These few runs do not establish which microbatch is best under sustained agent workloads; the actual input length remained 2,048 tokens.

Separately, an initial SWE-bench Pro v2 Hard task using the baseline, IQ2 SSD streaming and mini-SWE-agent 2.4.6 exceeded its 16,384-token context when the next request reached 20,546 tokens; it produced no valid score. A repeat with 131,072 positions allocated processed more than 22,000 tokens before being stopped to test this runtime candidate. It produced no submitted patch or score. Full DeepSWE and SWE-bench Pro Hard results, long-context retrieval and agent quality remain to be measured.
