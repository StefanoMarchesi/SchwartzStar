# Experimental patches

`build.sh` does not apply these. See the linked report before using one.

- `turbo3-v-cache.patch`: turbo3 value cache for the banded build. [Report](../results/turbo3-metal.md). The codec and the cache wiring are adapted from the TurboQuant llama.cpp fork [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) (MIT, llama.cpp and ggml authors), which follows the TurboQuant paper (arXiv 2504.19874). The port and the Metal kernels for `q4_0` keys with turbo3 values were written with AI assistance.
- `gpu-keepalive.patch`: keeps the GPU busy while the backend is computing, to reduce the SSD-streaming decode slowdown. Banded build, off by default (`GGML_METAL_KEEPALIVE=20000`). Not a new technique. [Report](../results/gpu-keepwarm.md). Written with AI assistance.
- `turbo3-banded.patch`: apply after `turbo3-v-cache.patch`; runs banded flash attention with turbo3 values on Metal. [Report](../results/turbo3-metal.md). Written with Freebuff (DeepSeek V4.1 Flash) and reviewed with another AI assistant; the report has the Freebuff disclosure.
- `turbo3-prefill.patch`: apply after the two turbo3 patches; faster turbo3 prompt processing on Metal, with `GGML_METAL_FA_KV_F16=0` to restore the earlier paths. [Report](../results/turbo3-metal.md). Written with Freebuff; the report has the disclosure.
- `turbo3-decode.patch`: apply after the three turbo3 patches; faster turbo3 decode on Metal. [Report](../results/turbo3-metal.md). Written with Freebuff; the report has the disclosure.
- `ssd-inplace-io.patch`: with `LLAMA_MOE_CACHE_INPLACE_IO=1`, SSD streaming reads router ids and writes the slot map in shared memory without GPU copies. Banded build, off by default. [Report](../results/ssd-inplace-io.md). Written with Freebuff; the report has the disclosure.
- `ssd-direct-slot-map.patch`: with `LLAMA_INKLING_SSD_DIRECT=1`, SSD streaming writes the expert slot map straight to shared memory instead of queueing a GPU copy. Banded build, off by default; no reliable speed gain shown. [Report](../results/direct-slot-map.md). Written with AI assistance.
