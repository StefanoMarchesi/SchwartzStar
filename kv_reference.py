#!/usr/bin/env python3
"""CPU reference for exact blockwise attention over a disk-backed KV stream."""
import math
import random
import struct
import tempfile


def read_rows(file, row_format, page_rows):
    page_bytes = row_format.size * page_rows
    while page := file.read(page_bytes):
        yield row_format.iter_unpack(page)


def online_attention(query, pages, bias, sink):
    # Include the virtual attention sink once. Its value vector is zero.
    maximum, weight_sum = sink, 1.0
    accumulator = [0.0] * len(query)
    scale = 1.0 / math.sqrt(len(query))
    position = 0
    for page in pages:
        scores_and_values = []
        for row in page:
            key, value = row[:len(query)], row[len(query):]
            score = math.fsum(q * k for q, k in zip(query, key)) * scale + bias(position)
            scores_and_values.append((score, value))
            position += 1
        page_max = max(score for score, _ in scores_and_values)
        new_max = max(maximum, page_max)
        history = math.exp(maximum - new_max)
        weight_sum *= history
        accumulator = [old * history for old in accumulator]
        for score, value in scores_and_values:
            weight = math.exp(score - new_max)
            weight_sum += weight
            for i, v in enumerate(value):
                accumulator[i] += weight * v
        maximum = new_max
    return [numerator / weight_sum for numerator in accumulator]


def verify():
    rng = random.Random(103)
    width, tokens, page_rows = 8, 4099, 257
    query = [rng.uniform(-1, 1) for _ in range(width)]
    row_format = struct.Struct(f"<{2 * width}f")
    # Synthetic position-dependent bias and a non-zero sink.
    def bias(position):
        return math.sin(math.log1p(tokens - position) * 4) * 5
    sink = 1.25
    with tempfile.TemporaryFile() as f:
        for _ in range(tokens):
            f.write(row_format.pack(*(rng.uniform(-2, 2) for _ in range(2 * width))))
        f.seek(0)
        streamed = online_attention(query, read_rows(f, row_format, page_rows), bias, sink)
        f.seek(0)
        rows = list(row_format.iter_unpack(f.read()))
    scores = [math.fsum(q * k for q, k in zip(query, row[:width])) / math.sqrt(width) + bias(i)
              for i, row in enumerate(rows)]
    maximum = max(sink, max(scores))
    weights = [math.exp(score - maximum) for score in scores]
    total = math.fsum(weights) + math.exp(sink - maximum)
    dense = [math.fsum(w * row[width + i] for w, row in zip(weights, rows)) / total for i in range(width)]
    delta = max(abs(a - b) for a, b in zip(streamed, dense))
    if delta > 1e-12:
        raise RuntimeError(f"delta max {delta:.3g}")
    print(f"{tokens} rows across SSD-style pages, relative bias and one sink: max delta {delta:.3g}")


if __name__ == "__main__":
    verify()
