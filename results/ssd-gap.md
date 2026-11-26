# Where SSD streaming loses time against resident weights

Apple M1 Pro, 16 GB, 26 November 2026. Banded build, Inkling 2 Mini UD-IQ2_XXS, flash attention on, `q4_0` keys and values. The machine had its usual applications open. These are small local measurements. They explain where to look next; they are not a speedup.

Across every comparison below, token IDs and log-probabilities did not change.

## Raw SSD reads

Uncached `pread` of random ranges of the GGUF (`F_NOCACHE`, 3 GiB per run, median of 3).

| Read size | 1 worker | 4 workers | 8 workers | 16 workers |
| ---: | ---: | ---: | ---: | ---: |
| 1.2 MiB (about one expert) | 2.11 | 4.85 | 4.80 | 5.02 |
| 4.8 MiB | 3.87 | 7.36 | 8.77 | 8.63 |

GiB/s. At expert size the drive is already saturated with the 4 workers the runtime uses. In a simulation with scattered destinations (`preadv`), merging only adjacent experts gained 8 to 12%. Bridging gaps gave from 38% slower to 24% faster and read more bytes.

## Prompt processing

At 4,096 positions, three prompts of about 1,900 tokens, two SSD runs: SSD took 73 to 82 s against 46 s resident. The runtime's blocked-read timer accounts for 93 to 101% of that difference.

A selection trace of a 7,895-token prompt (microbatch 256, 3,072 MiB expert cache) showed that a layer application selects 64% of its 128 experts on average, and about 62% of the selected experts are not cached.

### Reading the next layer ahead (prototype, not shipped)

While the GPU computed a layer, an experimental patch read the non-cached experts of the next layer into a staging buffer. Same binary, 7,895 tokens, runs in the order off, on, off, on:

| Read ahead | Prompt time s | Blocked on experts s | Peak footprint MiB |
| --- | ---: | ---: | ---: |
| off | 112.3 | 38.7 | 3378 |
| on | 111.8 | 29.2 | 3973 |
| off | 133.2 | 49.7 | 3365 |
| on | 112.0 | 25.6 | 3973 |

Blocked time fell, but total time did not improve reliably. Only 54% of the 283 GiB read ahead was used, and the rest of the run got slower. Competition for memory bandwidth is a likely reason, not a measured one. The patch used about 600 MiB more and is not included.

## Decode

Temporary probes recorded the GPU start and end time of every Metal command buffer, on the same clock as the expert cache callback. Per generated token, about 190 tokens:

| | Resident | SSD streaming |
| --- | ---: | ---: |
| Wall time | 66.7 ms | 205.9 ms |
| GPU busy (union of command buffers) | 58.3 ms | 120.3 ms |
| GPU idle | 8.4 ms | 85.6 ms |
| Command buffers | 2.9 | 195 |

In the SSD run, expert reads took 31.5 ms and the router read-back wait 16.5 ms per token. The GPU was idle for 49.5 ms while the cache callback ran and for 36 ms outside it. The graph is cut into about 50 splits, against 2 resident. GPU busy time also doubled for the same model math. Many short command buffers with idle gaps, for example through lower GPU clocks, could explain it. This has not been verified: clock speed was not measured.

Turning off the extra command buffer per graph (`n_cb` 0 instead of 1) cut buffers from 195 to 146 and did not help: 258 ms against 226 ms per token in back-to-back runs. The same setting measured 206 ms in an earlier run, so run-to-run noise is about 20 ms.

A control that copied cache misses from host memory instead of reading the SSD was not usable on 16 GB: keeping the experts resident caused heavy memory pressure.

### Cutting the resident graph into pieces

Later the same day: Metal concurrency or graph optimization turned off in resident mode left GPU busy time at 55 to 59 ms per token, so losing them does not explain the 120 ms.

A temporary scheduler probe then cut each resident graph split into pieces of 230 nodes, waited for each piece, and optionally slept before the next one. The same math produced the same tokens.

| Resident run | Command buffers per token | GPU busy ms per token | GPU idle ms per token |
| --- | ---: | ---: | ---: |
| normal | 2.9 | 57.0 | 20.5 |
| cut, no added pause | 94 | 93.2 | 64.5 |
| cut, 0.5 ms pause | 93 | 129.5 | 131.0 |
| cut, 1.5 ms pause | 99 | 153.6 | 202.8 |

GPU busy time rose with fragmentation and rose further as the pauses grew. This reproduces the SSD effect without any SSD read: the GPU does the same work more slowly when it arrives in small pieces with gaps. Lower GPU clocks during gaps would explain the second step; clock speed was not measured. One run per row. The first normal run was stopped by the swap guard, and the normal row is the repeat at the end of the series.

This points at the pauses and the number of command buffers as the target for faster SSD decode, rather than at the Metal kernels.
