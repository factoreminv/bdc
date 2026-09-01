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


def verify(manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    root = manifest_path.parent
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
    verify(args.manifest)
    print("upper certificate wrapper: ACCEPT")
