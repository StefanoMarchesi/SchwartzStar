#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

profile="${1:-quality}"
reasoning_default=0
case "$profile" in
    coding)   file=Inkling-2-Mini-UD-IQ2_XXS.gguf; ctx=8192; batch=512; cache=3072; reasoning_default=128 ;;
    resident) file=Inkling-2-Mini-UD-IQ2_XXS.gguf; ctx=2048; batch=16; cache=0 ;;
    minimum)  file=Inkling-2-Mini-UD-IQ1_S.gguf;   ctx=1024; batch=1; cache=512 ;;
    quality)  file=Inkling-2-Mini-IQ4_NL.gguf;     ctx=1024; batch=2; cache=768 ;;
    *) echo 'Usage: ./serve.sh [coding|quality|minimum|resident] [model.gguf]' >&2; exit 2 ;;
esac
if (( $# > 2 )); then
    echo 'Usage: ./serve.sh [coding|quality|minimum|resident] [model.gguf]' >&2
    exit 2
fi

bin="${INKLING_SERVER:-${INKLING_BUILD_DIR:-$PWD/build}/bin/llama-server}"
model="${2:-$PWD/models/$file}"
ctx="${INKLING_CTX:-$ctx}"
batch="${INKLING_BATCH:-$batch}"
cache="${INKLING_CACHE_MIB:-$cache}"
port="${INKLING_PORT:-8080}"
threads="${INKLING_THREADS:-4}"
cache_type_k="${INKLING_CACHE_TYPE_K:-f16}"
checkpoints="${INKLING_CTX_CHECKPOINTS:-2}"
read_cache="${INKLING_SSD_READ_CACHE:-0}"
reasoning_budget="${INKLING_REASONING_BUDGET:-$reasoning_default}"
gpu=all

for number in "$ctx" "$batch" "$cache" "$port" "$threads" "$checkpoints"; do
    [[ "$number" =~ ^(0|[1-9][0-9]*)$ ]] || { echo "Invalid numeric value: $number" >&2; exit 2; }
done
if (( ctx < 32 || batch < 1 || batch > ctx || port < 1 || port > 65535 || threads < 1 )); then
    echo 'Check the context, microbatch, port, and thread count.' >&2
    exit 2
fi
if [[ "$read_cache" != 0 && "$read_cache" != 1 ]]; then
    echo 'INKLING_SSD_READ_CACHE must be 0 or 1.' >&2
    exit 2
fi
if [[ ! "$reasoning_budget" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo 'INKLING_REASONING_BUDGET must be a non-negative number of tokens.' >&2
    exit 2
fi
reasoning=off
reasoning_effort=none
if [[ "$reasoning_budget" != 0 ]]; then
    reasoning=on
    reasoning_effort=minimal
fi
# llama-server probes recurrent-state rollback with a two-token logical batch.
logical_batch=$batch
if (( logical_batch < 2 )); then
    logical_batch=2
fi
if [[ ! -x "$bin" || ! -s "$model" ]]; then
    printf 'Missing server binary or model.\nServer: %s\nModel: %s\n' "$bin" "$model" >&2
    exit 1
fi
if [[ "$(uname -s)/$(uname -m)" != Darwin/arm64 ]]; then
    [[ "$profile" == resident ]] || { echo 'SSD streaming requires Apple Silicon.' >&2; exit 2; }
    gpu=0
fi

export LLAMA_INKLING_SSD=0
export LLAMA_INKLING_SSD_PAGE_CACHE="$read_cache"
extra=()
if [[ "$profile" != resident ]]; then
    if (( cache == 0 )); then
        echo 'SSD streaming requires a positive expert cache size.' >&2
        exit 2
    fi
    export LLAMA_INKLING_SSD=1
    extra=(--inkling-ssd --split-mode none --cpu-moe --moe-cache-mib "$cache" --moe-cache-cpu-share 0)
fi

exec "$bin" --model "$model" --alias inkling-2-mini --host 127.0.0.1 --port "$port" \
    --fit off --gpu-layers "$gpu" --no-op-offload --load-mode mmap --no-repack \
    --ctx-size "$ctx" --batch-size "$logical_batch" --ubatch-size "$batch" --threads "$threads" \
    --parallel 1 --no-warmup --cache-ram 0 --no-cache-idle-slots --no-ui --cors-origins "http://127.0.0.1:$port,http://localhost:$port" \
    --ctx-checkpoints "$checkpoints" \
    --flash-attn off --cache-type-k "$cache_type_k" --cache-type-v f16 \
    --verbosity 4 --reasoning "$reasoning" --reasoning-effort "$reasoning_effort" --reasoning-budget "$reasoning_budget" \
    ${extra[@]+"${extra[@]}"}
