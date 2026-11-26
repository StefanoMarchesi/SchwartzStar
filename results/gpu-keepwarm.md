# Keeping the GPU busy during SSD-streaming decode

Apple M1 Pro, 16 GB, 26 November 2026. Experimental: nothing in `build.sh` or `serve.sh` uses it.

## Latest: built into the server, longer runs

26 November 2026, later the same day. `patches-experimental/gpu-keepalive.patch` moves the load into the Metal backend (banded build sources). It is off by default, runs only with `GGML_METAL_KEEPALIVE=20000` (loop iterations per buffer), and parks after 200 ms without GPU work (`GGML_METAL_KEEPALIVE_HOLD_MS`). After a generation, an idle server used 0.8% to 2% CPU with it on, and its trace showed the thread parked.

About 1,900 prompt tokens, one prompt, **1,792 generated tokens** per run, 4,096 context, microbatch 256, K and V `q4_0`, 3,072 MiB expert cache for SSD streaming. Each run is a new server. SSD runs went off, on, on, off, on, off, off, on; this order and the comparison of adjacent pairs were fixed before the run. Two resident runs were interleaved (a third, and two planned runs with a lighter load, were not run).

| Setup | Decode tok/s, mean | Range | Prefill tok/s | Mac power, W | Joules per token |
| --- | ---: | --- | --- | ---: | ---: |
| Resident weights (2 runs) | 14.94 | 14.80 to 15.08 | 118, 132 | 52.2 | 3.50 |
| SSD streaming (4 runs) | 6.22 | 5.86 to 6.65 | 37 to 85 | 30.7 | 4.95 |
| SSD streaming, GPU kept busy (4 runs) | 8.18 | 7.59 to 8.52 | 71 to 84 | 34.5 | 4.24 |

- The four pairs gained **+28%, +37%, +24% and +38%** (median +32%). Decode with SSD streaming went from 2.4 to 1.8 times slower than resident weights.
- **All ten runs produced the same 1,792 tokens and log-probabilities**, resident or streamed, with the load on or off (one prompt).
- Power is whole-machine input power from the Mac's battery controller, on mains with a full battery. It includes the display and other apps, and each run has only 1 to 4 one-minute averages. Energy per token with the load on divided by off was 0.87, 0.70, 1.11 and 0.81 across the pairs. No clear energy cost per token appeared; the data are too noisy to claim a saving. Schiltmans measured a large cost at a small cache on an M4 Pro with another power meter, see [prior work](#prior-work).
- The gain is smaller than in the earlier series below (+47% to +142%), which used 128 generated tokens, the unmodified server and the separate helper. SSD streaming without the load was also faster here. The cause of the difference was not isolated.
- The Mac was also used for light browser work during part of the series.

## First result, with the separate helper

Running a small separate GPU load next to the server made SSD-streaming decode **47% to 142% faster** in four pairs of runs (median 71%), with identical output. The load uses power for as long as it runs. Power and temperature were not measured.

## Prior work

**Neither the slowdown nor the remedy is new.** These tests were run before we looked for earlier work, and an earlier version of this note did not credit it.

- [Alex Schiltmans's per-layer expert streaming](https://github.com/alexschiltmans/llama.cpp-per-layer-experts/blob/per-layer-experts/docs/moe-ssd-streaming.md) for llama.cpp, which this project already builds on for SSD streaming, has `LLAMA_MOE_CACHE_GPU_KEEPALIVE`: a dummy kernel on a private command queue while missed experts are read. The switch was documented on 24 October 2026. On an M4 Pro it reports gains from 9% to 77% depending on cache size, the GPU clock rising from about 580 to 1,580 MHz at 4,096 MiB, and the energy per token at that size rising about five times with it and its direct-ids switch together.
- [Mference PR 48](https://github.com/NeelM0906/Mference/pull/48) keeps the GPU clocked up between short decode command buffers with a one-SIMD-group spin on a second queue, stopping shortly after the last token.
- [Mference PR 54](https://github.com/NeelM0906/Mference/pull/54) turned that filler off on M1-series GPUs: on an M1 Max, decode fell from about 35 to about 3 tok/s with it. On the M1 Pro used here the load helped. The two cases differ in software, workload and chip, and we have not tested why. Do not assume it helps on every Mac.

What this note adds is a measurement on an M1 Pro with Inkling 2 Mini, and a per-command-buffer breakdown of one decode.

## Where the time went

An instrumented copy of the banded build logged every Metal command buffer during decode. Per generated token, without any change:

- about 192 command buffers: 96 for model computation, 48 to read back the selected expert ids and 47 to upload the slot map;
- the 95 copies kept the GPU busy for less than 2 ms in total;
- the GPU spent about 100 to 150 ms per token with nothing running. Between those pauses it waits for the CPU. Missing experts are read from the SSD straight into shared memory without a GPU copy; in one run the server's own counters showed about 30 ms of SSD reads per token;
- the computation itself took about 120 to 160 ms of GPU time. Resident weights needed about 58 ms in an [earlier measurement](ssd-gap.md).

[Fragmenting the resident graph](ssd-gap.md) had already reproduced slower GPU work without any SSD. If pauses make the GPU slow down, keeping it active should undo that.

## Probe with the instrumented build

`tools/gpu_keepwarm.swift` submits short compute buffers (about 0.6 to 0.8 ms, one group of 32 threads) in a loop from its own process. Order off, on, on, off, 64 tokens on each of three prompts:

| Per token | Off | On | On | Off |
| --- | ---: | ---: | ---: | ---: |
| GPU time of the server's work, ms | 158 | 60 | 59 | 131 |
| Time without server GPU work, ms | 150 | 82 | 71 | 104 |
| Total, ms | 308 | 142 | 130 | 235 |

With the extra load, GPU time for the server's own work came back to about the resident figure. The waiting time between GPU submissions also fell. These GPU times count only the server's command buffers; the keep-alive work runs on the same GPU. This fits the GPU lowering its performance state during pauses, but clock speed and power state were not measured.

## Confirmation with the unmodified banded build

Eight servers in the order off, on, on, off, on, off, off, on; about 1,900 prompt tokens, 128 generated on each of three prompts, 4,096 context, microbatch 256, K and V `q4_0`, 3,072 MiB expert cache, greedy. The order and the comparison of adjacent pairs were fixed before this series. Speeds are total tokens over total time for each server.

| Pair | Decode off, tok/s | Decode on, tok/s | Change | Prefill off / on, tok/s |
| --- | ---: | ---: | ---: | --- |
| 1 | 2.76 | 6.68 | +142% | 55.6 / 61.3 |
| 2 | 2.67 | 4.34 | +62% | 54.0 / 61.7 |
| 3 | 4.02 | 7.20 | +79% | 71.4 / 64.9 |
| 4 | 5.10 | 7.52 | +47% | 74.5 / 74.9 |
| Mean of 4 servers | 3.64 | 6.43 | | 63.9 / 65.7 |

The Mac was in use during this series (video encoding and a browser were seen running), and swap grew by up to 3.8 GB in one server. That probably explains part of the spread between pairs. The last pair had stable swap and equal prompt processing, and gave the smallest gain, +47%. Across all servers the slowest run with the load (4.34) was below the fastest without it (5.10), but every adjacent pair favoured the load. All eight servers produced the same tokens and log-probabilities (384 tokens each). Peak footprint of the server stayed between 3,295 and 3,309 MiB. Prompt processing did not change clearly.

## Limits

- One machine, on mains power, short prompts. Long contexts, long generations and other Macs were not tested.
- Energy use, temperature, fan noise and throttling over long sessions were not measured. Expect higher power draw, which matters on battery.
- The load also runs while the server is idle. Stop it when you are not generating.
- Only one load size was tested. A smaller or intermittent load might give the same gain with less power.
- The effect on other GPU work, such as video or games, was not measured.

## Trying it

In one terminal:

```bash
swiftc -O tools/gpu_keepwarm.swift -o gpu-keepwarm
```

```bash
./gpu-keepwarm
```

Then run the server in another terminal as usual and stop the load with Ctrl-C when done. The published file adds handling for more stop signals and comments to the measured one; the GPU work is the same.
