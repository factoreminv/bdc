# Self-contained logarithm enclosure

The exhaustive Bellman verifier does not call NumPy `log2`, the C library `log2`, MPFR, or
another transcendental implementation. `upper/src/ivl.py` encloses every logarithm using
exact range reduction, exact-rational table construction, a positive series remainder, and
outward binary64 arithmetic.

## Mathematical enclosure

For every positive binary64 input `x`, `numpy.frexp` supplies the exact decomposition

```text
x = 2^e m,       1 <= m < 2.
```

Let `N = 2^14`. The implementation selects

```text
j = floor(N(m-1)),       c = 1 + j/N,
z = (m-c)/(m+c).
```

At an exact table boundary it deliberately selects the preceding cell. Thus
`0 <= m-c <= 1/N` and

```text
0 <= z <= 1/(2N).
```

For `|z|<1`, the atanh series gives

```text
ln(m/c) = 2 sum_{q>=0} z^(2q+1)/(2q+1).
```

Every term is nonnegative. Retaining only `2z` and bounding all later denominators below by
three gives

```text
2z <= ln(m/c) <= 2z + 2 z^3 / (3(1-z^2)).
```

The code uses the universal rational upper bound obtained by substituting `z=1/(2N)` in
the remainder. Therefore no sampled or asymptotic error estimate enters the enclosure.

## Table and `1/ln(2)`

Each table center `c=1+j/N` is rational. At import time the code encloses `ln(c)` by the same
atanh identity, retaining terms through `q=30` and adding

```text
2 z^63 / (63(1-z^2)).
```

All of this preprocessing uses Python `Fraction`, so the bounds are exact rationals. The
same construction at `c=2` encloses `ln(2)`, after which reciprocal endpoints enclose
`1/ln(2)`. Conversion of each rational endpoint to binary64 is checked with
`Fraction.from_float`; `nextafter` is applied whenever the nearest float lies on the wrong
side.

At runtime the verifier combines

```text
log2(x) = e + log2(c) + ln(m/c)/ln(2)
```

using its ordinary outward operations. The integer exponent is added to the table interval
before the residual to avoid avoidable cancellation near `x=1`.

## Trusted arithmetic

The remaining numerical assumptions are the ordinary ones used by the rest of the Bellman
replay: IEEE-754 binary64 round-to-nearest, preserved subnormals, and no unsafe reassociation
or fast-math transformation. `frexp` is used only for exact binary exponent extraction;
`nextafter` is used for endpoint movement. There is no transcendental-library accuracy
assumption.

## Regression tests

Run

```sh
(cd upper && python -m src.test_log2_enclosure)
```

The test compares the enclosure with 220-bit `mpmath` at every one of the 16384 table
boundaries, adjacent binary64 values, several exponent extremes, and 100000 random points.
This catches implementation regressions. It is not a premise of the proof above.
