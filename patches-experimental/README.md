# Experimental patches

`build.sh` does not apply these. See the linked report before using one.

- `turbo3-v-cache.patch`: turbo3 value cache for the banded build. [Report](../results/turbo3-metal.md). The codec and the cache wiring are adapted from the TurboQuant llama.cpp fork [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) (MIT, llama.cpp and ggml authors), which follows the TurboQuant paper (arXiv 2504.19874). The port and the Metal kernels for `q4_0` keys with turbo3 values were written with AI assistance.
- `ssd-direct-slot-map.patch`: with `LLAMA_INKLING_SSD_DIRECT=1`, SSD streaming writes the expert slot map straight to shared memory instead of queueing a GPU copy. Banded build, off by default; no reliable speed gain shown. [Report](../results/direct-slot-map.md). Written with AI assistance.
