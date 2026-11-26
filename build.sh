#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# "banded" is experimental: Unsloth's Metal banded attention (PR 431) plus the same patches.
variant="${INKLING_VARIANT:-stable}"
patches=("$PWD"/patches/*.patch)
case "$variant" in
    stable)
        revision=ae6a7ad2c5754dd8db93d04c84fda70435864f6e
        remote=https://github.com/ggml-org/llama.cpp.git
        default_source="$PWD/.llama.cpp"
        default_build="$PWD/build" ;;
    banded)
        revision=09df2193f242c5b578265fafda55d118e2cc1730
        remote=https://github.com/unslothai/llama.cpp.git
        default_source="$PWD/.llama.cpp-banded"
        default_build="$PWD/build/banded"
        patches+=("$PWD"/patches-banded/*.patch) ;;
    *) echo 'INKLING_VARIANT must be stable or banded.' >&2; exit 2 ;;
esac
remote="${INKLING_REMOTE:-$remote}"
source_dir="${INKLING_SOURCE_DIR:-$default_source}"
build_dir="${INKLING_BUILD_DIR:-$default_build}"
case "$(uname -s)/$(uname -m)" in
    Darwin/arm64) metal=ON ;;
    Linux/*) metal=OFF ;;
    *) echo 'Use macOS on Apple Silicon, or Linux for CPU checks.' >&2; exit 2 ;;
esac

if [[ ! -d "$source_dir/.git" ]]; then
    git init "$source_dir"
fi
if ! git -C "$source_dir" rev-parse --verify HEAD >/dev/null 2>&1; then
    git -C "$source_dir" fetch --depth 1 "$remote" "$revision"
    git -C "$source_dir" checkout --detach FETCH_HEAD
fi
if [[ "$(git -C "$source_dir" rev-parse HEAD)" != "$revision" ]]; then
    echo "Unexpected llama.cpp revision in $source_dir" >&2
    exit 1
fi

# Build the expected patched tree without touching the checkout.
patch_index=$(mktemp "${TMPDIR:-/tmp}/inkling-index.XXXXXX")
rm "$patch_index"
trap 'rm -f "$patch_index" "$patch_index.lock"' EXIT
export LC_ALL=C
GIT_INDEX_FILE="$patch_index" git -C "$source_dir" read-tree HEAD
for patch in "${patches[@]}"; do
    GIT_INDEX_FILE="$patch_index" git -C "$source_dir" apply --cached "$patch"
done
expected=$(GIT_INDEX_FILE="$patch_index" git -C "$source_dir" write-tree)
if [[ -z "$(git -C "$source_dir" status --porcelain)" ]]; then
    GIT_INDEX_FILE="$patch_index" git -C "$source_dir" diff --cached --binary |
        git -C "$source_dir" apply --index
elif ! git -C "$source_dir" diff --quiet || [[ -n "$(git -C "$source_dir" ls-files --others --exclude-standard)" ||
        "$(git -C "$source_dir" write-tree)" != "$expected" ]]; then
    echo "Unexpected changes in $source_dir. Use a dedicated checkout." >&2
    exit 1
fi
cmake -S "$source_dir" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release \
    -DGGML_METAL="$metal" -DLLAMA_OPENSSL=OFF \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_APP=OFF \
    -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
cmake --build "$build_dir" --config Release --target llama-server --parallel "${JOBS:-4}"
if [[ "$metal" == ON ]]; then
    xcrun clang -O2 -Wall -Wextra "$PWD/footprint.c" -o "$build_dir/bin/inkling-footprint"
fi
