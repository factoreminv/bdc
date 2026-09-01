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

## Python environment

Python 3.11 or later is recommended. From the archive root, create an isolated environment
and install the listed packages:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

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
cd upper
python -m src.verify_rigor_certificate results/rigor_d0.65_L6_k6_m30_manifest.json
python -m src.test_rigor_certificate
```

The complete Bellman calculation evaluates both actions at all `2^30` states. It is
computationally substantial and uses approximately the elapsed time reported in the manifest
on the original machine:

```sh
BDC_NP=12 BDC_M1F=14 \
BDC_BIAS=results/bias_d0.65_L6_k6_m28.npy \
python -m src.audit_rigor 13 20 6 6 30 results/q_d0.65_L6_k6_kraft.npz
```

The final printed normalized upper endpoint must not exceed `0.250984`. The potential is an
arbitrary bounded function in the Bellman inequality; its role is to sharpen the bound, not
to add an assumption.

The cutting-plane and value-iteration programs are search procedures: they produced the
fixed code lengths and potential supplied here. The proof uses only the fixed Kraft and
Bellman inequalities. It does not assume that either search converged or found an optimum.

The Bellman replay uses NumPy binary64 interval endpoints. Elementary operations are widened
as specified in `upper/src/ivl.py`, and reordered reductions receive a Higham error bound.
The separate Kraft check uses 300-bit `mpmath`. Entropy logarithms in the Bellman pass use
NumPy `log2` widened by four ulps. Accordingly, the result relies on an IEEE-754
round-to-nearest environment without fast-math, unsafe reassociation, or flush-to-zero, and
on a four-ulp error bound for the platform `log2`. The function `validate_log2` compares
selected inputs against 200-bit `mpmath`; this is a diagnostic, not a proof of the library
contract. The original manifest does not record a complete compiler and libm identity.

## Lower bound

The following command recomputes the entropy and segmentation terms with 192-bit Arb balls
and exact integer-polynomial convolutions:

```sh
cd lower
python run_rigorous_final.py
python verify_rigorous_certificate.py out/rigorous_lower_certificate.json
python test_rigorous_certificate.py
```

The generator overwrites `out/rigorous_lower_certificate.json`. Its final lower endpoint must
be at least `0.12415`. Truncated positive contributions are omitted, while subtracted tails
are bounded from above, so every rounding and truncation weakens the claimed lower bound.

## Lean 4

With Lean and Lake installed:

```sh
cd lean
lake exe cache get
lake build Bdcproof.Cert
```

The project contains no `sorry`. Lean proves the Bellman telescoping implication, a geometric
tail inequality, an abstract upper-bound assembly lemma, the monotonicity-extension algebra
conditional on the cited monotonicity result, preservation of lower bounds under retained
nonnegative corrections, and the final lower-bound decimal arithmetic. It does not formalize
the channel models, entropy identities, filter-state or segmentation reductions, numerical
libraries, exhaustive enumeration, Arb computation, or stored data. Thus it is not an
end-to-end formalization of either capacity bound.

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
