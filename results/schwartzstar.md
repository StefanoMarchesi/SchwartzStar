# SchwartzStar on a 16 GB M1 Pro

Measurements from the internal SSD of MacBookPro18,1 on macOS 27.3 beta, 24 November 2026. One model server at a time, one slot, four threads, Metal. The runtime is llama.cpp at `ae6a7ad2c5754dd8db93d04c84fda70435864f6e` with this repository’s patches. The [release notes](../RELEASE-v0.1.0-alpha.md) pin the GGUFs and their hashes.

## Runtime measurements

The 512-token trials use the same tokenized prefix and generate 24 tokens. Each row comes from one fresh process. Cache capacity and microbatch size are reported together because both changed between the two streaming configurations.

| Mode | Context | Microbatch | Expert cache | Prefill | First text | Decode | Footprint | Peak RSS | End-to-end swap change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IQ3 resident | 4,096 | 64 | — | 58.08 tok/s | 8.84 s | 4.63 tok/s | 680 MiB | 7,462 MiB | +1,683 MiB |
| IQ3 SSD | 4,096 | 16 | 3,456 MiB | 14.02 tok/s | 36.53 s | 3.64 tok/s | 4,002 MiB | 5,081 MiB | −32 MiB |
| IQ3 SSD | 4,096 | 128 | 3,072 MiB | 38.78 tok/s | 13.21 s | 3.31 tok/s | 3,636 MiB | 4,725 MiB | −40 MiB |
| IQ3 SSD | 4,096 | 512 | 3,072 MiB | 66.44 tok/s | 7.71 s | 3.42 tok/s | 3,638 MiB | 4,763 MiB | −7 MiB |
| IQ2 SSD | 4,096 | 128 | 3,072 MiB | 56.42 tok/s | 9.08 s | 4.86 tok/s | 3,666 MiB | 4,705 MiB | 0 MiB |
| IQ2 SSD with system page cache | 4,096 | 128 | 3,072 MiB | 67.73 tok/s | 7.56 s | 5.15 tok/s | 3,667 MiB | 4,719 MiB | 0 MiB |
| IQ2 SSD | 4,096 | 512 | 3,072 MiB | 87.26 tok/s | 5.87 s | 4.60 tok/s | 3,666 MiB | 4,704 MiB | −8 MiB |
| IQ2 resident | 4,096 | 128 | — | 101.98 tok/s | 5.03 s | 16.09 tok/s | 671 MiB | 6,048 MiB | 0 MiB |

Each streaming run covered 48/48 layer applications. Prefill reads from the GGUF fell from 106,686.97 MiB at u16 to 46,315.27 MiB at u128. The reported physical footprint is the process high-water mark. File-backed mappings and macOS’s cache contribute to system memory pressure even when they contribute little to that metric. The resident run reported 11,293 MiB of Metal buffer allocations, including 10,863 MiB of model buffers. Background memory changed during all trials.

At the same cache capacity, IQ2 requested 21,575.31 MiB in prefill at u128 and 8,821.41 MiB at u512. The two u128 page-cache configurations requested the same number of bytes and emitted the same text over 24 generated tokens. Their timing is one sequential trial. In page-cache mode macOS may satisfy reads from memory.

The original cache allocation failed with IQ3 u8 and u16 because a mixed quantization layout with relatively few layers received too few slots. Patch 0008 gives each layout enough slots for one worst-case routed microbatch. Patch 0009 lets the SSD path use a larger microbatch when the slots fit. The shared buffers, placement of the routed weights, and 48-application coverage are checked on startup or in the benchmark log.

```bash
IQ3=$(./download.sh iq3)
INKLING_CTX=4096 INKLING_BATCH=128 INKLING_CACHE_MIB=3072 \
  python3 check.py run quality "$IQ3" iq3-ssd-p512 --prefix-only 512 --tokens 24 --timeout 180
```

## 8,000-token retrieval probe

IQ2 SSD with 8,192 configured positions, microbatch 256, 2,304 MiB of expert cache, Q4_0 keys and F16 values. The test planted three distinct inventory codes at token offsets 1,000, 4,000, and 7,000. The model evaluated all **8,000** input tokens and returned **all three codes correctly** in a constrained JSON response. First text arrived in **118.09 s**, prefill reached **67.75 tok/s**, and 38 decoded tokens averaged **4.02 tok/s**. Peak process footprint was **3,137 MiB**, peak RSS **3,951 MiB**, and the process covered 48/48 layer applications. System swap fell by about 120 MiB during the run from a pre-existing 2,280 MiB.

```bash
IQ2=$(./download.sh iq2)
INKLING_CTX=8192 INKLING_BATCH=256 INKLING_CACHE_MIB=2304 INKLING_CACHE_TYPE_K=q4_0 INKLING_REASONING_BUDGET=0 \
  python3 check.py run coding "$IQ2" iq2-8k-retrieval --prefix-only 8000 --retrieve --tokens 96 --timeout 600
```

The probe checks exact recovery on one synthetic prefix. The generated object contained `alder = RV7KQ3`, `birch = J9MD4X`, and `cedar = PT5N8B`.

### Comparison with resident Gemma at 8K and 16K

The same raw Completion probe was run on Gemma 4 12B Instruct Q4_0 at 8,000 and 16,000 evaluated tokens and repeated on IQ2 at 16,000. Both used one slot, four threads, microbatch 256, Q4_0 keys, F16 values, and the same three-code JSON schema. Codes occupy the same relative offsets. Each model tokenizes the repeated filler with its own tokenizer. Inkling stops after 38 generated tokens at both lengths. The original Inkling 8K run had a 96-token cap, with 64 for the other runs.

| Model | Input / context | Exact codes | Prefill | Decode | First text | Peak RSS | Peak physical footprint | Peak swap increase |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Inkling IQ2 SSD | 8,000 / 8,192 | 3/3 | 67.75 tok/s | 4.02 tok/s | 118.09 s | 3,951 MiB | 3,137 MiB | 0 MiB |
| Gemma Q4_0 resident | 8,000 / 8,192 | 0/3 | 144.33 tok/s | 14.43 tok/s | 55.45 s | 5,709 MiB | 1,208 MiB | 1,338 MiB |
| Inkling IQ2 SSD | 16,000 / 16,384 | 3/3 | 59.97 tok/s | 4.07 tok/s | 266.83 s | 4,109 MiB | 3,695 MiB | 0 MiB |
| Gemma Q4_0 resident | 16,000 / 16,384 | 0/3 | 135.31 tok/s | 13.12 tok/s | 118.26 s | 8,024 MiB | 1,502 MiB | 85 MiB |

In the 16K trial, IQ2 recovered all three codes with 49% lower peak RSS. Gemma processed the input and generated text faster. The macOS physical-footprint metric orders the models differently because of how it charges file-backed mappings and private buffers. System swap measures activity across the laptop and starting conditions varied. These results cover a single structured retrieval scenario on raw completion with quantized keys. Chat formatting, the default F16 keys, and long-context code edits need separate evaluation.

```bash
INKLING_CTX=16384 INKLING_BATCH=256 INKLING_CACHE_MIB=2304 INKLING_CACHE_TYPE_K=q4_0 INKLING_REASONING_BUDGET=0 \
  python3 check.py run coding "$IQ2" iq2-16k-retrieval --prefix-only 16000 --retrieve --tokens 64 --timeout 600
INKLING_CTX=16384 INKLING_BATCH=256 INKLING_CACHE_TYPE_K=q4_0 \
  python3 check.py run resident "$GEMMA_MODEL" gemma-16k-retrieval --prefix-only 16000 --retrieve --tokens 64 --timeout 600
```

`check.py` saves the JSON result before returning a failure status when any code misses.

## 128K allocation probe

IQ4_NL, Q4_0 keys and F16 values, microbatch 1, expert cache 256 MiB. The server allocated **131,072** positions and evaluated **32** input tokens. It emitted eight tokens, with first text in **14.76 s**, prefill **2.17 tok/s**, decode **2.28 tok/s**, and all **48/48** routed applications observed. Peak physical footprint was **3,175 MiB**, peak RSS **4,907 MiB**, and swap changed from 2,292.1 to 2,284.1 MiB. The Metal KV buffers allocated 2,624 MiB for the eight global applications and 128.12 MiB for the local window.

The 32-token input tests allocation and short inference. Retrieval and inference with 128K populated remain open. The current attention graph builds a relative-bias tensor whose size grows with both the microbatch and the populated keys. The upstream experimental Metal banded attention proposal targets that problem. [Metal banded Flash Attention proposal](https://github.com/unslothai/llama.cpp/pull/431)

```bash
IQ4=$(./download.sh iq4)
INKLING_CTX=131072 INKLING_BATCH=1 INKLING_CACHE_MIB=256 INKLING_CACHE_TYPE_K=q4_0 \
  python3 check.py run quality "$IQ4" iq4-q4k-128k-start --prefix-only 32 --tokens 8 --timeout 90
```

## Coding benchmark

The [coding agent report](coding-agent.md) tracks native tool calling. The guided PHP probe uses the same two source files and tests for both models. The model emits JSON edits. The controller reads and writes only the two fixture sources and runs each test under a macOS sandbox. The second turn also reruns the first test. Allowlist, fixture, and tool logs are in the [probe source](../tools/php_patch_probe.py).
