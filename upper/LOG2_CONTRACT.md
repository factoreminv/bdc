# The platform `log2` contract

The converse verifier evaluates entropies with NumPy's `log2` and widens each result
outward by `LOG2_ULP = 4` (`upper/src/ivl.py`). The interval enclosure is therefore
rigorous **relative to an assumption**: that the platform `log2` returns a value within
that widening of the exact result, for every argument reached during the run.

This file records what is assumed, what has been measured, and what would be needed to
remove the assumption. Nothing here changes a constant or an inequality.

## What is assumed

`ivl.log2` computes `y = np.log2(x)` and returns
`[y - 4(|y|2^-52 + 10^-300), y + 4(|y|2^-52 + 10^-300)]`.
The enclosure is valid whenever the platform `log2` has relative error at most
`4 * 2^-52` on the arguments actually evaluated. NumPy delegates to the platform libm;
the common implementations (glibc, Apple libm, and the SIMD paths NumPy may select)
are documented or measured as faithful (error `<= 1` ulp), which the widening covers
with margin — but *documented as faithful* is a vendor statement about a build, not a
proof about the binary that ran, and NumPy may dispatch to a vectorized path whose
accuracy differs from the scalar one.

`ivl.validate_log2` samples ~4000 arguments across four adversarial regimes (near 1,
near 0, uniform, and exponentially spread) and compares against 200-bit `mpmath`. On
the machine recorded in `environment/verified-2026-09-01.json` it reported
0 enclosure failures and a worst relative deviation of `1.07e-16` (about `0.48` ulp),
consistent with a faithful implementation. **Sampling ~4000 of the arguments in a run
that evaluates on the order of `10^12` is a diagnostic, not a proof.**

## The size of the problem

Measured by instrumenting `ivl` and counting elements passed to `np.log2` in
`IvModel.rbar_ub` for both actions:

| quantity | value |
| --- | --- |
| `log2` evaluations per state (both actions) | 1,280 |
| states in the certified run | `2^30` |
| `log2` evaluations in a full replay | `1.374 x 10^12` |

## Measured throughput of candidate replacements

Apple M-series, Python 3.12.9, NumPy 2.5.2, python-flint 0.9.0, mpmath 1.4.1
(`/tmp/bench_log2.py`, single core; the last column divides by 12 processes and adds
the non-`log2` remainder of the archived 7,996 s interval pass):

| implementation | rate (eval/s) | guarantee | est. full-replay wall time |
| --- | --- | --- | --- |
| NumPy `log2` (vectorized) | `2.27 x 10^8` | none; assumed `<= 4` ulp | 7,996 s (measured, archived) |
| python-flint Arb `log`, 64-bit (scalar) | `1.81 x 10^6` | proved ball enclosure | ~20 h (estimate) |
| mpmath 200-bit (scalar) | `1.45 x 10^5` | correct to working precision | ~9 days (estimate) |

The wall-time column is an **estimate**, not a measurement: it assumes the surrounding
interval arithmetic stays vectorized and only the logarithm becomes a scalar loop, and
it ignores the per-call Python overhead of marshalling `10^12` values into and out of
Arb. Treat it as a lower bound on the cost of that route.

## Assessment

Arb is the substantive candidate, and the interesting finding is that it is *not*
obviously out of reach: a proved enclosure at roughly an order of magnitude more wall
time than the archived run. `arb_log` returns a ball guaranteed to contain the exact
logarithm, so it would replace the four-ulp assumption with a library-level theorem
rather than a vendor accuracy claim.

Two routes were considered and not taken here:

- **Arb per-element.** Correct in principle; needs the scalar-loop integration to be
  written and benchmarked in place rather than in isolation, because the estimate above
  omits marshalling cost.
- **A correctly rounded vectorized `log2`** (for example CORE-MATH's, which carries a
  proof of correct rounding). This is the route that would preserve throughput, but it
  requires building the library, pinning the build configuration, binding it to NumPy,
  and establishing that the proof applies to the compiled binary. It was not benchmarked.

Neither was integrated, and no replay was run with either. **The four-ulp contract
therefore stands as an explicit, unresolved assumption of the converse**, recorded here,
in `README.md`, and in the environment record. It is not disguised as a proved property,
and the converse should be read as rigorous *relative to it*.

## What would close it

1. Integrate a logarithm with a documented rounding theorem.
2. Record the exact library version and build configuration in the manifest.
3. Rerun the full `2^30`-state Bellman pass with it.
4. Require the normalized endpoint to remain at most `0.250984`.

Until all four are done, this assumption should be stated wherever the converse is
claimed to be machine-checked.
