#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
case "${1:-iq4}" in
    iq1) file=Inkling-2-Mini-UD-IQ1_S.gguf
         sha=17a7ff34acb0e1fe878309962458e0633172c7f93ffe3b1c4f5f6b317dc317db
         size=7764293056 ;;
    iq2) file=Inkling-2-Mini-UD-IQ2_XXS.gguf
         sha=2f953a44281cf861b19ccfdb2ce65f7648e1e92bda4af3729893604fd907c544
         size=8486237632 ;;
    q2)  file=Inkling-2-Mini-UD-Q2_K_XL.gguf
         sha=b42841608ea380ddde5a3f1125081357128a3c71b80d006246de4e31292fc6c1
         size=9138716608 ;;
    iq3) file=Inkling-2-Mini-UD-IQ3_XXS.gguf
         sha=fd9ccedae17c1bb89744612fbb46730b5275f2a342133d0419776bd1ae311657
         size=11663310784 ;;
    iq4) file=Inkling-2-Mini-IQ4_NL.gguf
         sha=cbdf3fa9dc50a4124977980f37da4f1fa764b0cfe800d275cd43d26ce2a81af4
         size=17627940768 ;;
    *) echo 'Usage: ./download.sh [iq1|iq2|q2|iq3|iq4]' >&2; exit 2 ;;
esac
if [[ "$file" == *-IQ4_NL.gguf ]]; then
    repo=ggml-org/Inkling-2-Mini-GGUF
    revision=e8a2cecf0e78914f07cc27b9be711a9b2c7561d9
else
    repo=unsloth/Inkling-2-Mini-GGUF
    revision=8078d149286eec7193795413b9e0643da6d11fa1
fi
mkdir -p models
path="models/$file"
verify() {
    test "$(wc -c < "$1" | tr -d ' ')" = "$size" &&
        printf '%s  %s\n' "$sha" "$1" | shasum -a 256 -c - >&2
}
if [[ -f "$path" ]]; then
    verify "$path" || { echo "Existing file failed verification: $path" >&2; exit 1; }
else
    lock="$path.lock"
    mkdir "$lock" || { echo "A download is active or a previous run left a lock: $lock" >&2; exit 1; }
    trap 'rmdir "$lock"' EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    if [[ ! -f "$path" ]]; then
        part="$path.part"
        if [[ ! -f "$part" ]] || [[ "$(wc -c < "$part" | tr -d ' ')" != "$size" ]]; then
            curl -fL --retry 3 --continue-at - --output "$part" \
                "https://huggingface.co/$repo/resolve/$revision/$file?download=true" >&2
        fi
        verify "$part" || { echo "Download incomplete or hash mismatch: $part" >&2; exit 1; }
        mv "$part" "$path"
    fi
fi
printf '%s\n' "$PWD/$path"
