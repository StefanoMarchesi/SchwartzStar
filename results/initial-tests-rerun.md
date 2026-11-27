# Rerunning the first tests with the experimental options

Apple M1 Pro, 16 GB, 27 November 2026. UD-IQ2_XXS, SSD streaming. This repeats the first tests with their original settings. Only the tests that needed no new downloads were rerun: the [HumanEval+ smoke](humaneval-smoke.md), the [Flask case](swebench-lite-smoke.md) and the [mini-SWE-agent fixture](mini-swe-agent-smoke.md) were not.

## Configurations

- **A:** the published stable build, as in the first tests.
- **B:** the banded build with the [GPU keep-alive](gpu-keepwarm.md) and [in-place ids and slot map](ssd-inplace-io.md) patches, both on (`INKLING_GPU_KEEPALIVE=1 INKLING_INPLACE_IO=1`). In the 8K tests the banded attention is not used (flash attention off in the PHP test; the retrieval test uses the same profile). A and B therefore differ in build as well as in the two options.
- **C:** B plus `INKLING_NGRAM=1` ([n-gram report](ngram-speculative.md)).
- **D:** B with the `long` profile (32K configured, banded flash attention, `q4_0` K and V, microbatch 512).

No run was flagged as disturbed. There are at most two runs per configuration, fewer than the four pairs this project requires before claiming a speed gain.

## Retrieval of three codes (cold prompt, 38 generated tokens)

| Prompt | Config | Codes | Prompt processing, tok/s | First text, s | Decode, tok/s | Peak footprint, MiB |
| --- | --- | --- | --- | --- | --- | --- |
| 8,000 | A (2 runs) | 3/3, 3/3 | 58.7, 57.1 | 136.2, 140.2 | 3.96, 4.09 | 3,142, 3,137 |
| 8,000 | B (2 runs) | 3/3, 3/3 | 63.8, 63.9 | 125.5, 125.2 | 7.47, 7.65 | 3,136, 3,138 |
| 16,000 | A | 3/3 | 55.5 | 288.1 | 3.98 | 3,400 |
| 16,000 | B | 3/3 | 60.4 | 265.0 | 7.26 | 3,248 |

The first run of this test on 24 November reported 67.75 tok/s prompt processing and 118 s to first text at 8,000 tokens with the stable build, faster than today's A, so absolute prompt speeds vary between days.

## Guided two-turn PHP task, 8K

Every run passed both turns. Times are seconds until the tests passed.

| Config | Turn 1 | Turn 2 | Decode, tok/s | Peak footprint turn 2, MiB |
| --- | --- | --- | --- | --- |
| A | 69.2, 60.2 | 96.9, 90.0 | 4.2 to 4.9 | 4,290, 4,289 |
| B | 40.5, 37.9 | 53.0, 52.4 | 8.2 to 9.1 | 4,296, 4,294 |
| C (n-gram) | 47.0, 46.3 | 67.8, 65.6 | 6.6 to 7.2 | 4,452, 4,642 |
| D (long, 32K) | 36.8, 36.8 | 54.8, 52.6 | 8.8 to 9.2 | 3,787, 3,784 |

- A and B produced the same code changes in both turns. C and D produced a slightly different second-turn change that also passed.
- **N-gram lookup made this task slower.** Only 48 of 314 drafted tokens were accepted in turn 1, and 80 of 456 in turn 2. It helps when the output copies long stretches of earlier text, as in the [code copy test](ngram-speculative.md), not in this short guided edit.
- Servers were ready in 1.8 to 3.3 s with the model file already read once before; the very first retrieval run took 6.7 s.

## Compared with Thinking Machines' published results

[Thinking Machines' announcement](https://thinkingmachines.ai/news/inkling-2-mini/) and [model card](https://thinkingmachines.ai/model-card/inkling-2-mini/) report suites such as SWE-Bench, DeepSWE and LongBench for the model in its native NVFP4/MXFP8 precision. The pages read did not state temperature, context length or, for most suites, the reasoning setting, and gave no measured generation speed. **None of the tests here is one of those suites, and the weights here are a 2-bit third-party quantization, so no comparison is made.**
