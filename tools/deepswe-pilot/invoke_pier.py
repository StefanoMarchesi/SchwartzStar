#!/usr/bin/env python3
"""Invoke Pier's installed console entry point with this interpreter.

The package does not provide ``python -m pier``. Resolving the console script
entry point through importlib metadata binds Pier to the selected interpreter
without trusting a separately located executable or its shebang.
"""
from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, distribution


def main() -> int:
    try:
        entries = distribution("datacurve-pier").entry_points
    except PackageNotFoundError as exc:
        raise SystemExit("datacurve-pier distribution is not installed") from exc
    entry = next((item for item in entries if item.group == "console_scripts" and item.name == "pier"), None)
    if entry is None:
        raise SystemExit("datacurve-pier has no console_scripts entry point named 'pier'")
    app = entry.load()
    result = app()
    return 0 if result is None else int(result)


if __name__ == "__main__":
    raise SystemExit(main())
