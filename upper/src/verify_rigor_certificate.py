"""Verify the immutable wrapper around the exhaustive upper-bound run.

This inexpensive checker authenticates every input and output, checks the Kraft
condition independently at high precision, and checks that the archived outward
endpoint implies the advertised decimal.  Replaying the Bellman maximum itself is
done by ``audit_rigor.py`` using the command recorded in the manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import numpy as np

from src.kraft import check as kraft_check


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


RELEASE_ASSETS = {
    "bias_d0.65_L6_k6_m28.npy": (
        "This 1 GiB Bellman potential exceeds the GitHub limit on repository content "
        "and ships as an asset of the v1.0 release.  From the repository root:\n"
        "    gh release download v1.0 -R factoreminv/bdc \\\n"
        "      -p bias_d0.65_L6_k6_m28.npy -D upper/results/"
    ),
}


def require_inputs(root: Path, names) -> None:
    """Report every absent input at once, with retrieval instructions where they apply.

    A fresh clone does not contain the release asset, so this is the first thing a
    reader hits; an unadorned FileNotFoundError from the hashing loop does not say
    which file is meant or where to obtain it.
    """
    missing = [name for name in names if not (root / name).is_file()]
    if not missing:
        return
    lines = ["missing certificate input(s) under %s:" % root]
    for name in missing:
        lines.append("  - %s" % name)
        hint = RELEASE_ASSETS.get(name)
        if hint:
            lines.append("    " + hint.replace("\n", "\n    "))
    raise FileNotFoundError("\n".join(lines))


def verify(manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    root = manifest_path.parent
    require_inputs(root, manifest["files"])
    for name, expected in manifest["files"].items():
        actual = sha256(root / name)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {name}: {actual} != {expected}")

    qpath = root / manifest["test_measure"]
    nctx, nbad, worst, *_ = kraft_check(qpath, prec=300)
    if nbad:
        raise ValueError(f"Kraft violation in {nbad}/{nctx} contexts; worst={worst:+.3e}")

    result = np.load(root / manifest["result_archive"])
    theta = float(result["theta_hi"])
    if theta != float(manifest["theta_hi_over_one_minus_d"]):
        raise ValueError("archived theta does not equal the manifested theta")
    advertised = Fraction(manifest["theorem_decimal"])
    if Fraction.from_float(theta) > advertised:
        raise ValueError("outward upper endpoint exceeds advertised theorem decimal")
    if int(manifest["states_checked"]) != 1 << int(manifest["m"]):
        raise ValueError("state-count metadata is inconsistent")
    if int(manifest["actions_per_state"]) != 2:
        raise ValueError("binary action count must be two")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    # verify() keeps raising, so the adversarial tests still see the exception; the
    # command line reports the absent input as a message rather than a traceback.
    try:
        verify(args.manifest)
    except FileNotFoundError as exc:
        raise SystemExit("upper certificate wrapper: MISSING INPUT\n%s" % exc)
    print("upper certificate wrapper: ACCEPT")
