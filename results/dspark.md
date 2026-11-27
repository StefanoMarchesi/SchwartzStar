# DSpark drafter with SSD streaming

Apple M1 Pro, 16 GB, 27 November 2026. Experimental. DSpark itself, its llama.cpp support and the drafter GGUF are not ours: this only measures them with our SSD streaming.

## Setup

- llama.cpp `ca123ba` with patches 0001 to 0010 ([upstream check](upstream-ca123ba.md)) and `patches-experimental/ssd-dspark-drafter-resident.patch`. Without that one-line patch our SSD streaming switch also applies to the drafter, which is not a looped Inkling model, and loading fails. Not the banded build, and without the keep-alive or in-place patches.
- Drafter: `dspark/dspark-Inkling-2-Mini-BF16-mainline.gguf_file` from `unsloth/Inkling-2-Mini-GGUF`, revision `8078d149` (749 MB, SHA-256 `dba3d73cd7cb342460e0c12f3d46f29eb4641ad13b87526b8f262a5fa2cd157b`). It stays in memory.
- `--spec-type draft-dspark` with its default of 3 drafted tokens, chosen before the series. Same prompts and method as the [n-gram test](ngram-speculative.md): 4,096 context, 384 greedy tokens, order off, on, on, off, on, off, off, on on one binary. A run during which another process passed the script's load threshold was repeated once.

## Results

| Request | Off, tok/s | On, tok/s | On / off per pair | Drafted / accepted |
| --- | --- | --- | --- | --- |
| Code edit | 3.99, 4.09, 4.14, 4.07 | 9.21, 7.87, 7.89, 7.52 | 2.31, 1.92, 1.91, 1.85 | 289 / 286 |
| Story | 4.55, 5.09, 5.98, 5.41 | 6.13, 5.48, 5.43, 5.45 | 1.35, 1.08, 0.91, 1.01 | 598 / 183 |

- **Code edit:** faster in all four pairs, with the same 384 tokens. The last repeated run was still disturbed.
- **Story:** no consistent gain (median +4%, one pair 9% slower). The text diverged at token 33, where the two top candidates were close; before that, the chosen token's log-probability differed by up to 0.26. As the [n-gram report](ngram-speculative.md#rejected-drafts-and-exactness) shows, batching alone can move this quantized model's probabilities; no separate control was run for this build.
- Peak process footprint rose by 136 to 156 MiB (about 3.30 to 3.45 GiB).
- Across both requests, decode read 21% more from the SSD with the drafter (79,230 against 65,428 MiB) while processing 54% more tokens, rejected drafts included.

## Limits

Two prompts, a short context, greedy decoding. The speeds are not comparable with the n-gram results, which used another build with two extra experimental patches. Other numbers of drafted tokens were not tried.
