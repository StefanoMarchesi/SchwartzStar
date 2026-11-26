# Keeping the GPU busy during SSD-streaming decode

Apple M1 Pro, 16 GB, 26 November 2026. Experimental: nothing in `build.sh` or `serve.sh` uses it.

## Result

Running a small separate GPU load next to the server made SSD-streaming decode **47% to 142% faster** in four pairs of runs (median 71%), with identical output. The load uses power for as long as it runs. Power and temperature were not measured.

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
