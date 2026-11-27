# N-gram speculative decoding with SSD streaming

Apple M1 Pro, 16 GB, 27 November 2026. No code change: this measures the n-gram lookup speculation already built into llama.cpp (`--spec-type ngram-simple`) with Inkling 2 Mini and SSD streaming. Nothing here is a new technique.

## Setup

Banded build with the [GPU keep-alive](gpu-keepwarm.md) and [in-place ids and slot map](ssd-inplace-io.md) experimental patches, both on in every run. UD-IQ2_XXS, 4,096 context, microbatch 256, K and V `q4_0`, 3,072 MiB expert cache. Raw `/completion`, greedy, 384 tokens per request. One server per run, order off, on, on, off, on, off, off, on, with default n-gram sizes (12 and 48) chosen before the series. Speed is time per emitted token.

- **Code edit:** copy a repetitive Python module (947 prompt tokens) with one method renamed. This is close to the best case for n-gram lookup.
- **Story:** continue a short story opening.

| Request | Off, tok/s | On, tok/s | On / off per pair | Drafted / accepted |
| --- | --- | --- | --- | --- |
| Code edit | 6.72, 7.45, 7.48, 7.47 | 15.35, 16.98, 17.59, 17.70 | 2.29, 2.28, 2.35, 2.37 | 388 / 351 |
| Story | 7.44, 8.59, 8.64, 8.75 | 8.28, 9.65, 9.72, 9.96 | 1.11, 1.12, 1.12, 1.14 | 217 / 97 |

- The code edit gave the same 384 tokens off and on.
- The story diverged at token 216, at a near tie between two candidates, so after that point the speeds are not measured on the same text.
- Across both requests, the runs with drafting processed 32% more tokens (rejected drafts included) and read only 5% more from the SSD during decode.
- Short lookups made it slower. With sizes 2 and 8, single runs gave 8.4 tok/s on the code edit and 6.3 on the story. With 1 and 4, they gave 4.6 and 5.3.
- Peak memory and checkpoint size were not measured. The machine was in use, with no other process above the script's 150% CPU threshold when checked before and after each run.

## Rejected drafts and exactness

Inkling 2 has recurrent state, so the server saves a checkpoint, restores it after a rejected draft and replays the accepted tokens.

- **Code edit:** the output stayed identical to the run without drafting even with the short sizes, with up to about 340 drafted tokens rejected.
- **Story:** it diverged at token 111 (sizes 2 and 8) and 13 (1 and 4).
- **Control without drafting or restore:** the same prefix from the run without drafting was processed in batches of 1, 8 and 256 tokens. This alone moved the next-token distribution by up to 0.14 in total variation, and by up to 0.86 in log-probability, over the shared top five candidates. The runs with drafting reached 0.13 and 0.64, slightly above the control at a few positions. With a batch of 256 at the first position, where the grouping matches the original run, the match was exact. After many rejections the difference did not accumulate.

So **output with drafting is not guaranteed to be bit-identical.** How tokens are grouped into batches is enough to explain the differences seen, and no restore error was observed. A small restore error hidden below that noise is not ruled out. The control also means that this quantized model on Metal can give a different greedy text just because the prompt was split differently.

## Limits

Two prompts, one context size, greedy only, not tested with long contexts, with the default non-banded build, or with a drafter model. Real coding work will usually repeat less than the code edit used here.
