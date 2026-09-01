# Execution environments

This directory separates two things that must not be confused: the environment that
**produced** the stored certificates, and environments in which those certificates have
since been **checked**.

## The original environment is not recorded

`upper/results/rigor_d0.65_L6_k6_m30_manifest.json` records the parameters, the state
count, the elapsed time, and the SHA-256 of every input and output. It does **not**
record the Python, NumPy, SciPy, mpmath, python-flint/FLINT, operating system, CPU,
compiler, or libc/libm identity of the machine that ran the 2^30-state Bellman pass, and
those identities cannot be recovered from the archived artifacts: the `.npz` members
carry NumPy's zeroed ZIP timestamps, and nothing else in the repository names a host.

No attempt has been made to guess them. If the original machine or its logs are still
available, the missing identities should be added here as a separate record, clearly
labelled as the original environment.

## Environments in which the checks have passed

`verified-2026-09-01.json` is a full record of one machine on which the short checks were
run and passed, produced by `tools/print_environment.py`. `requirements-tested-2026-09-01.txt`
pins the exact package versions of that environment.

On that machine the lower generator reproduced every mathematical field of the shipped
`lower/out/rigorous_lower_certificate.json` bit for bit, on a NumPy and python-flint
substantially newer than any that could have produced it. Only `elapsed_seconds` differed.
That is evidence of the lower calculation's portability; it is not a statement about the
upper Bellman pass, which was not rerun.

## Recording the environment of a future replay

Run this immediately before a long calculation and keep the output beside the result:

```sh
python tools/print_environment.py -o environment/replay-$(date +%Y-%m-%d).json
```

The record includes empirical probes of the arithmetic contract the upper bound depends
on: round-to-nearest-even, preserved subnormals, no unsafe reassociation, and a sampled
comparison of the platform `log2` against 200-bit `mpmath`. These probes are diagnostics.
Passing them does not prove the contract holds for every argument reached in a
2^30-state replay; see the arithmetic paragraph in the top-level `README.md`.
