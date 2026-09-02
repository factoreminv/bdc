"""Print, and optionally save, the identity of the environment running a replay.

The stored certificates were produced on a machine whose software and hardware identity
was not recorded (see ``environment/README.md``).  This script exists so that no future
replay repeats that omission: run it immediately before a long calculation and keep its
output beside the result.

    python tools/print_environment.py                       # print
    python tools/print_environment.py -o env.json           # print and save

The floating-point section matters because the upper-bound interval arithmetic requires
IEEE-754 binary64 with round-to-nearest-even, no unsafe reassociation or fast-math
contraction, and preserved subnormals.  The verifier's logarithm is now a self-contained
range-reduced rational-series enclosure and does not call the platform ``log2``.  The checks
below are diagnostics of the implementation and execution environment; the analytic
enclosure proof is recorded in the manuscript and top-level README.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import sysconfig
from datetime import datetime, timezone


def _version(mod: str) -> str | None:
    try:
        m = __import__(mod)
    except Exception:
        return None
    return getattr(m, "__version__", None) or getattr(m, "version", None)


def _flint_identity() -> dict:
    out: dict = {"python_flint": _version("flint")}
    try:
        import flint
        for attr in ("__flint_version__", "flint_version", "__version_info__"):
            if hasattr(flint, attr):
                out["flint_library"] = str(getattr(flint, attr))
                break
    except Exception:
        pass
    return out


def _libc_libm() -> str | None:
    """Identify the C library supplying libm.

    platform.libc_ver() reads an ELF binary and returns empty strings on macOS, where
    libm lives in libSystem; report the OS build there instead, since that is what
    pins the Apple libm in use.
    """
    ver = " ".join(x for x in platform.libc_ver() if x)
    if ver:
        return ver
    if platform.system() == "Darwin":
        rel, _, _ = platform.mac_ver()
        return f"Apple libSystem (libm) on macOS {rel}" if rel else "Apple libSystem (libm)"
    return None


def _fp_probes() -> dict:
    """Empirical probes of the arithmetic contract."""
    import numpy as np

    probes: dict = {}

    # Round-to-nearest, ties-to-even: 1 + 2^-53 must round back to 1.0, while
    # 1 + 2^-52 must be the next representable value.
    one = np.float64(1.0)
    probes["round_to_nearest_even"] = bool(
        one + np.float64(2.0) ** -53 == one
        and one + np.float64(2.0) ** -52 > one
    )

    # Subnormals preserved (no flush-to-zero): the smallest positive subnormal must
    # survive a store and a multiply.
    tiny = np.float64(5e-324)
    probes["subnormals_preserved"] = bool(tiny > 0 and (tiny * np.float64(2.0)) > tiny)

    # No unsafe reassociation of a sum whose exact value is 0 only under reordering.
    a = np.array([1.0, 1e16, -1e16], dtype=np.float64)
    probes["sum_not_reassociated"] = bool(float(a[0] + a[1] + a[2]) == 0.0)

    # Diagnostic comparison of the self-contained log2 enclosure with 200-bit mpmath.
    try:
        sys.path.insert(0, "upper")
        from src.ivl import validate_log2  # type: ignore
        bad, worst = validate_log2(np.random.default_rng(0), ntest=20000)
        probes["log2_implementation"] = "rational-table plus bounded atanh residual"
        probes["log2_enclosure_failures"] = int(bad)
        probes["log2_worst_relative_deviation"] = float(worst)
        probes["log2_note"] = (
            "sampled diagnostic only; soundness comes from the rational-series remainder "
            "and outward binary64 operations, not from this comparison"
        )
    except Exception as exc:  # pragma: no cover - diagnostic only
        probes["log2_probe_error"] = repr(exc)
    finally:
        if sys.path and sys.path[0] == "upper":
            sys.path.pop(0)
    return probes


def collect() -> dict:
    import numpy as np

    record = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "record_kind": "observed-on-this-machine",
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "compiler": platform.python_compiler(),
            "executable": sys.executable,
        },
        "packages": {
            "numpy": _version("numpy"),
            "scipy": _version("scipy"),
            "mpmath": _version("mpmath"),
            **_flint_identity(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "libc_libm": _libc_libm(),
            "config_platform": sysconfig.get_platform(),
        },
        "numpy_build": {
            "blas_openblas_or_accelerate": None,
        },
        "floating_point": _fp_probes(),
        "threading_env": {},
    }

    import os
    for var in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
        "BDC_NP", "BDC_M1F", "BDC_BIAS",
    ):
        record["threading_env"][var] = os.environ.get(var)

    try:
        cfg = np.show_config(mode="dicts")  # numpy >= 1.25
        blas = cfg.get("Build Dependencies", {}).get("blas", {})
        record["numpy_build"]["blas_openblas_or_accelerate"] = blas.get("name")
        record["numpy_build"]["blas_version"] = blas.get("version")
        record["numpy_build"]["compilers"] = {
            k: v.get("version") for k, v in cfg.get("Compilers", {}).items()
        }
    except Exception:
        pass

    try:
        record["cpu_count"] = os.cpu_count()
    except Exception:
        pass

    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", help="also write the record to this JSON file")
    args = parser.parse_args()
    record = collect()
    text = json.dumps(record, indent=2, sort_keys=False)
    print(text)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text + "\n")
        print(f"\nwritten to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
