# Banded build: long-context measurements

Apple M1 Pro, 16 GB unified memory, internal SSD, 26 November 2026. Inkling 2 Mini UD-IQ2_XXS with SSD expert streaming, four threads, greedy sampling. The machine had its usual applications open and system swap was high and variable. Unless noted, each row is a single run: do not read small speed differences as effects of a setting.

The binary used for these measurements reports source `0fbd675`. That local snapshot no longer exists. A rebuild from public commit `09df2193f242c5b578265fafda55d118e2cc1730` with the project patches and the NVFP4 patch (SHA-256 `52c96a21daf3759eb177d42c13d404de77cd931b4523df5099d01468357208ec`) produced identical token IDs and log-probabilities in the compared runs: two alternating pairs with `q4_0`/`q4_0`, one `f16`/`f16` run and one run with flash attention off. Prompt speed in the second alternating pair was 72.4 tok/s (rebuild) and 73.9 tok/s (old binary).

## Planted-code recall

A synthetic operations log with eight six-character codes at different depths, asked back in a shuffled order. The response is constrained by a JSON schema. Microbatch 256.

| Prompt tokens | Attention, K / V | Documents | Codes | Expert cache MiB | Peak footprint MiB |
| ---: | --- | ---: | --- | ---: | ---: |
| about 7,900 | flash off, q4_0 / f16 | 3 | 24/24 | 2304 | not recorded |
| about 7,900 | flash on, f16 / f16 | 3 | 24/24 | 2304 | not recorded |
| about 7,900 | flash on, q4_0 / f16 | 3 | 24/24 | 2304 | not recorded |
| about 7,900 | flash on, q4_0 / q8_0 | 3 | 24/24 | 2304 | not recorded |
| about 7,900 | flash on, q4_0 / q4_0 | 3 | 24/24 | 2304 | not recorded |
| 15,898 | flash on, f16 / f16 | 1 | 8/8 | 2304 | 3222 |
| 15,898 | flash on, q4_0 / f16 | 1 | 8/8 | 2304 | 2969 |
| 15,898 | flash on, q4_0 / q8_0 | 1 | 8/8 | 2304 | 2804 |
| 15,898 | flash on, q4_0 / q4_0 | 1 | 8/8 | 2304 | 2715 |
| 31,883 | flash on, q4_0 / q4_0 | 1 | 8/8 | 3072 | 3611 |

Every configuration returned the same constrained text as `f16`/`f16` on the same document. The test hit its ceiling and cannot rank the cache formats. A repeated run returned identical token IDs and log-probabilities.

## Memory at larger allocations

Server start only, no inference, `q4_0`/`q4_0`, expert cache 2304 MiB.

| Positions | KV cache MiB | Metal compute buffer, microbatch 256 | microbatch 64 |
| ---: | ---: | ---: | ---: |
| 16,384 | 200 | 736 | 184 |
| 65,536 | 632 | 2536 | 634 |
| 131,072 | 1208 | 3201 | 1234 |

Allocation does not show that a populated context of that size works.

## Coding check and cache reuse

The [guided PHP probe](coding-agent.md) at 8,192 positions, microbatch 512, expert cache 3072 MiB and two context checkpoints passed both turns and the regression check on the first attempt:

| Attention, K / V | Turn 1 | Turn 2 | Peak footprint MiB |
| --- | ---: | ---: | ---: |
| flash on, q4_0 / q4_0 | 61.8 s | 78.9 s | 3512 |
| flash on, q4_0 / f16 | 50.6 s | 73.2 s | 3875 |
| flash off, f16 / f16 | 53.4 s | 90.1 s | 4290 |

At 16,384 positions with two checkpoints, an appended turn reused 15,922 tokens and took 4.6 s. The same document with a different question near the end reused none and took 252.9 s: both checkpoints sit at the end of the earlier prompt (tokens 15,612 and 15,864).

## End-to-end check

`INKLING_VARIANT=banded ./build.sh` on a clean checkout, then `./serve.sh long`, then the 31,883-token recall prompt: 8 of 8 codes, prompt 568.1 s (56.1 tok/s), decode 2.90 tok/s, peak footprint 3,722 MiB, same token IDs as the earlier run. The checkout was cloned from a local git bundle, so the network fetch in `build.sh` was not exercised.

Log-probabilities differed from the earlier run by up to 0.149 (4 of 92 steps above 0.01). Two 8K runs separate the causes. With zero checkpoints, the `build.sh` binary matched the old binary exactly. With the same binary, two checkpoints against zero kept the same token IDs but moved log-probabilities by up to 0.46. Checkpoints split the prompt into different microbatches, so results with and without them are not bit-identical.
