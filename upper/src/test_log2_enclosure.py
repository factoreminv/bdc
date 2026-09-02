"""Adversarial diagnostics for the self-contained log2 interval enclosure.

These comparisons are not the proof.  Soundness follows from the exact rational table and
positive series remainder in ivl.py.  The test guards the implementation at every table
boundary, neighboring binary64 values, exponent extremes used by the verifier, and random
points.
"""
from __future__ import annotations

import numpy as np
from mpmath import mp, mpf, log

from src.ivl import _TABLE_N, log2


def main() -> None:
    mp.prec = 220
    values: list[float] = []
    for j in range(_TABLE_N + 1):
        c = np.float64(1.0 + j / _TABLE_N)
        if c >= 2.0:
            c = np.nextafter(np.float64(2.0), -np.inf)
        values.extend((float(np.nextafter(c, -np.inf)), float(c),
                       float(np.nextafter(c, np.inf))))
    for e in (-996, -500, -53, -1, 0):
        scale = np.float64(2.0) ** e
        values.extend(float(np.float64(x) * scale) for x in values[:3 * (_TABLE_N + 1)])
    rng = np.random.default_rng(20260902)
    values.extend((2.0 ** rng.uniform(-996, 0, 100000)).tolist())
    x = np.asarray([v for v in values if 1e-300 <= v <= 1.0], dtype=np.float64)
    lo, hi = log2((x, x))
    ln2 = log(2)
    for i, v in enumerate(x):
        exact = log(mpf(float(v))) / ln2
        if not (mpf(float(lo[i])) <= exact <= mpf(float(hi[i]))):
            raise AssertionError(
                f"log2 enclosure failure at {v!r}: [{lo[i]!r},{hi[i]!r}] vs {exact}"
            )
    print(f"self-contained log2 diagnostics: PASS ({len(x)} values)")


if __name__ == "__main__":
    main()
