# Supplementary material for "New Bounds on the Binary Deletion Channel Capacity"

This archive contains the data and source code needed to reproduce the two numerical
inequalities in the paper, together with the Lean 4 statements of the finite lemmas used in
their assembly. The search programs that originally found the displayed input law, test
measure, and Bellman potential are not needed to check the results and are therefore omitted.

## Contents

- `upper/`: the fixed test measure and Bellman potential, the interval evaluator, the recorded
  result, and its manifest;
- `lower/`: the fixed 44-atom input law, the Arb/FLINT calculation, the recorded one-sided
  bounds, and consistency tests;
- `lean/`: the Lean 4 project containing the finite algebraic and order-theoretic lemmas.

The upper calculation checks the coefficient `0.250984` at deletion probability `13/20`.
The lower calculation checks the coefficient `0.12415` for the fixed Poisson-repeat input.

## Running the commands

Every command below is written to be run **from the repository root**. Blocks that need a
different working directory run in a subshell, so pasting the blocks in order leaves you at
the root throughout.

## Python environment

Python 3.11 or later is recommended. From the repository root, create an isolated environment
and install the listed packages:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` gives lower bounds only. `environment/requirements-tested-2026-09-01.txt`
pins the exact versions of an environment in which the short checks have been run and passed,
and `environment/README.md` explains what that record is — and, importantly, what it is not:
the environment that originally produced the stored certificates was not recorded and has not
been reconstructed.

## Upper bound

The Bellman potential `bias_d0.65_L6_k6_m28.npy` is 1.0 GiB, which exceeds the GitHub limit
on repository content, so it is attached to the `v1.0` release instead of being committed.
Both commands below need it in place:

```sh
gh release download v1.0 -R factoreminv/bdc \
  -p bias_d0.65_L6_k6_m28.npy -D upper/results/
```

Its SHA-256 is recorded in the manifest and checked by the wrapper below; to check it by
hand:

```sh
shasum -a 256 upper/results/bias_d0.65_L6_k6_m28.npy
```

```text
964ee2dd5df9209a2a455e41f2691c7ae163eec16e6fe6343dc59470e375c0bc
```

The short check authenticates the three fixed inputs, verifies the Kraft inequalities at
300-bit precision, and checks that the stored outward endpoint implies the theorem decimal.
It does not repeat the `2^30`-state Bellman calculation:

```sh
(cd upper && python -m src.verify_rigor_certificate results/rigor_d0.65_L6_k6_m30_manifest.json)
(cd upper && python -m src.test_rigor_certificate)
```

If the release asset is absent the wrapper stops with `upper certificate wrapper: MISSING
INPUT`, names the file, and repeats the download command, rather than raising a bare
`FileNotFoundError`.

The complete Bellman calculation evaluates both actions at all `2^30` states. It is
computationally substantial and uses approximately the elapsed time reported in the manifest
on the original machine:

```sh
(cd upper && BDC_NP=12 BDC_M1F=9 \
  BDC_BIAS=results/bias_d0.65_L6_k6_m28.npy \
  python -m src.audit_rigor 13 20 6 6 30 results/q_d0.65_L6_k6_kraft.npz)
```

The final printed normalized upper endpoint must not exceed `0.250984`. The potential is an
arbitrary bounded function in the Bellman inequality; its role is to sharpen the bound, not
to add an assumption.

### What the full replay needs

| resource | value | basis |
| --- | --- | --- |
| disk | 1.0 GiB for the release asset, plus ~0.5 MiB of repository content | measured |
| peak RAM | 6.07 GiB peak memory footprint; 4.30 GiB largest single process | **measured** on the verification machine (`/usr/bin/time -l`, 2026-09-01) |
| processes | 12 (`BDC_NP=12`), one per core | the archived command |
| wall time | **10,004 s** for the current self-contained-log replay; 19,502 s for the historical platform-log replay on the same 12-core Apple M-series | current manifest; replay records dated 2026-09-02 and 2026-09-01 |
| CPU time | 71,435 s user + 76,227 s system | measured; the large system share is page traffic against the 2 GiB shared potential, so the run is memory-bound and does not saturate 12 cores |

Elapsed time is not part of the claim. The smaller prefix block used by the current command
improves cache behavior and offsets part of the cost of the self-contained logarithm.

Set `BDC_NP` to the core count you have; fewer cores raise the wall time roughly in
proportion and lower the worker memory. `BDC_M1F` sets the leaf-batch size (`2^BDC_M1F`
states per unit of work) and trades memory against scheduling overhead.

Record the environment before starting, and keep it with the result:

```sh
python tools/print_environment.py -o environment/replay-$(date +%Y-%m-%d).json
```

Elapsed time and the order in which reductions are performed may differ from the archived
run; neither is part of the claim. What must hold is the one-sided conclusion: the final
normalized endpoint must be no larger than `0.250984`.

The cutting-plane and value-iteration programs are search procedures: they produced the
fixed code lengths and potential supplied here. The proof uses only the fixed Kraft and
Bellman inequalities. It does not assume that either search converged or found an optimum.

The Bellman replay uses NumPy binary64 interval endpoints. Elementary operations are widened
as specified in `upper/src/ivl.py`, and reordered reductions receive a Higham error bound.
The separate Kraft check uses 300-bit `mpmath`. Entropy logarithms do not call NumPy `log2`
or any platform transcendental. They use exact power-of-two range reduction, a 16384-entry
table generated from exact rational atanh-series bounds, and a one-term residual series with
an explicit positive remainder. `upper/LOG2_CONTRACT.md` gives the derivation. The result
therefore relies on IEEE-754 round-to-nearest without fast-math, unsafe reassociation, or
flush-to-zero, but no longer assumes an ulp bound for a platform `log2`. The mpmath comparison
in `python -m src.test_log2_enclosure` is a regression test, not a premise of soundness.

## Lower bound

The following command recomputes the entropy and segmentation terms with 192-bit Arb balls
and exact integer-polynomial convolutions:

```sh
(cd lower && python run_rigorous_final.py)
(cd lower && python verify_rigorous_certificate.py out/rigorous_lower_certificate.json)
(cd lower && python test_rigorous_certificate.py)
```

The generator overwrites `out/rigorous_lower_certificate.json`. Its final lower endpoint must
be at least `0.12415`. Truncated positive contributions are omitted, while subtracted tails
are bounded from above, so every rounding and truncation weakens the claimed lower bound.

The functions `correction_m4_arb` and `correction_t2_arb` compute segmentation-type
entropies. Their likelihood weights sum over the compatible multinomial allocations of an
observed count within each type. In the manuscript notation, the complete hidden state is
`(T,Q)`, the classical baseline accounts for the allocation term `H(Q|T,W,Z)`, and these
corrections retain a one-sided part of `H(T|W,Z)`. They must not be interpreted as direct
lower bounds on the entropy of the complete count-allocation state.

## Lean 4

With Lean and Lake installed:

```sh
(cd lean && lake exe cache get && lake build Bdcproof.Cert && lake build)
```

The project contains no `sorry`, and continuous integration enforces it: the Lean job audits
every declaration in the `Bdcproof` library and fails if any reaches an axiom outside `propext`,
`Classical.choice`, `Quot.sound`. Lean proves the Bellman telescoping implication, a
geometric tail inequality, an abstract upper-bound assembly lemma, the
monotonicity-extension algebra conditional on the cited monotonicity result, preservation of
lower bounds under retained nonnegative corrections, and the final lower-bound decimal
arithmetic. It does not formalize the channel models, entropy identities, filter-state or
segmentation reductions, numerical libraries, exhaustive enumeration, Arb computation, or
stored data. Thus it is not an end-to-end formalization of either capacity bound.

## Authenticating the whole supplement

`RELEASE_MANIFEST.json` records the SHA-256 of every tracked file, of the release asset,
and of the environment records, together with the stored endpoints and the standing
assumptions. Regenerate it after any change:

```sh
python tools/release_manifest.py -o RELEASE_MANIFEST.json
```

A second script cross-checks that this file, the two certificates, the Lean decimals, and
the constants quoted here still agree; continuous integration runs it on every push:

```sh
python tools/check_consistency.py
```

## Licence and citation

The contents are released under the MIT licence (`LICENSE`). `CITATION.cff` records the
paper title, author, repository URL, and version; it deliberately carries no DOI, arXiv
identifier, or publication status, because none is settled.

## Expected results

```text
upper certificate wrapper: ACCEPT
upper adversarial tests: PASS
lower certificate wrapper: ACCEPT
lower adversarial tests: PASS
```

The long upper calculation need not reproduce elapsed time or intermediate reduction order.
It must return an outward endpoint no larger than the theorem decimal. The lower calculation
may print slightly different midpoint formatting across compatible FLINT versions, but its
one-sided inequalities must imply the same decimal bound.
