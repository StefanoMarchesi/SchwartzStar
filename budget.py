#!/usr/bin/env python3
"""Estimate the KV payload for Inkling 2 Mini. No runtime is launched."""
import argparse
from fractions import Fraction

MIB = 1024 ** 2
GIB = 1024 ** 3
FULL_LAYERS = 8
SWA_LAYERS = 40
SWA_WINDOW = 1024
ELEMENTS_PER_LAYER_TOKEN = 2 * 8 * 128  # K + V, eight KV heads, head width 128
KV_FORMATS = {
    "fp16": Fraction(2),
    "q8_0": Fraction(34, 32),
    "q4_0": Fraction(18, 32),
    "turbo3*": Fraction(25, 64),  # Atomic Engine PR: ~800 MiB for global KV at 131072
    "2bit ideal*": Fraction(1, 4),
}


def kv_payload(tokens, full_format, local_format=None):
    local_format = full_format if local_format is None else local_format
    full = FULL_LAYERS * tokens * ELEMENTS_PER_LAYER_TOKEN * full_format
    local = SWA_LAYERS * min(tokens, SWA_WINDOW) * ELEMENTS_PER_LAYER_TOKEN * local_format
    return int(full), int(local)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tokens", nargs="*", type=int, default=[1024, 32768, 131072, 524288, 1048576])
    parser.add_argument("--resident-kv-mib", type=int, default=1024)
    parser.add_argument("--ssd-gib-s", type=float, default=3, help="Hypothetical sustained SSD bandwidth")
    args = parser.parse_args()
    if any(t < 1 for t in args.tokens) or args.resident_kv_mib < 0 or args.ssd_gib_s <= 0:
        parser.error("invalid budget or token count")
    print(f"{'tokens':>10} {'format':>12} {'global GiB':>11} {'local GiB':>10} {'cold GiB/tok':>13} {'I/O max tok/s':>14}")
    for tokens in args.tokens:
        for label, size in KV_FORMATS.items():
            full, local = kv_payload(tokens, size)
            cold = max(0, full - args.resident_kv_mib * MIB) / GIB
            ceiling = f"{args.ssd_gib_s / cold:.2f}" if cold else "-"
            print(f"{tokens:>10d} {label:>12} {full/GIB:>11.2f} {local/GIB:>10.2f} {cold:>13.2f} {ceiling:>14}")
    print("* Experimental extrapolation or ideal lower bound. Rows show payload without padding, scratch, recurrent state or weights.")
    print("I/O bound scans all cold global K/V per token. It excludes expert I/O, seeks, GPU compute and prefill.")


if __name__ == "__main__":
    main()
