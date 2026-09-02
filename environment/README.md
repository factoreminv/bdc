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

## The full replay has been reproduced once

`replay-2026-09-01.json` records a complete rerun of the `2^30`-state Bellman pass on the
machine described in that file, together with its result. The regenerated
`upper/results/rigor_d0.65_L6_k6_m30.npz` came out **byte-identical** to the archived one
(SHA-256 `9baef9f6...d93465`), so the exhaustive interval maximum reproduced exactly on
different hardware and a newer NumPy than could have produced it. Wall time differed by a
factor of 2.4 (19,502 s against the recorded 7,996 s); the claim does not depend on it.

This historical replay used the earlier platform-`log2` implementation. It established
reproduction of that archived calculation but did not discharge its four-ulp assumption.
The current verifier has since replaced that call by the self-contained rational-series
enclosure documented in `upper/LOG2_CONTRACT.md`; a new full replay must be recorded
separately rather than rewriting this historical environment record.

## Full replay with the self-contained logarithm

`replay-self-contained-log-2026-09-02.json` records the replacement full replay on the same
12-core Apple M-series machine. It evaluated both actions at all `2^30` states in 10,004 s
and produced the normalized outward endpoint `0.250983755329845`, strictly below the theorem
decimal `0.250984`. The result archive has SHA-256
`ce529f06f2cbaf7d7b11a3d13a13f54d5d363462445fc12dbb99fe570ef86fc0`.
This is the current certificate. Unlike the historical replay, its certified entropy
calculation does not call a platform transcendental function.

## Recording the environment of a future replay

Run this immediately before a long calculation and keep the output beside the result:

```sh
python tools/print_environment.py -o environment/replay-$(date +%Y-%m-%d).json
```

The record includes empirical probes of round-to-nearest-even, preserved subnormals, and no
unsafe reassociation, together with a comparison of the self-contained logarithm enclosure
against 200-bit `mpmath`. These probes are diagnostics. The logarithm proof itself is the
exact rational construction and remainder bound in `upper/LOG2_CONTRACT.md`.
