"""Authenticate and check the outward-rounded lower certificate endpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
from decimal import Decimal, getcontext
from pathlib import Path

getcontext().prec = 80


def ball_lower(text: str) -> Decimal:
    # python-flint prints ``[mid +/- radius]``.  The stored midpoint is already
    # the lower endpoint because run_rigorous_final serializes x.lower().
    body = text.strip()[1:-1]
    return Decimal(body.split("+/-", 1)[0].strip())


def verify(path: Path) -> None:
    cert = json.loads(path.read_text())
    law = path.parent / cert["law_file"]
    actual = hashlib.sha256(law.read_bytes()).hexdigest()
    if actual != cert["law_sha256"]:
        raise ValueError("run-law SHA-256 mismatch")
    b = {k: Decimal(v) for k, v in cert["lean_component_bounds"].items()}
    for key in ("rdm_lower", "m4_lower", "t2_lower", "D_lower", "nu_lower"):
        if b[key] > ball_lower(cert[key]):
            raise ValueError(f"Lean lower endpoint for {key} is not conservative")
    if b["L_upper"] < ball_lower(cert["L_upper"]):
        raise ValueError("Lean upper endpoint for L is not conservative")
    assembled = b["rdm_lower"] + b["nu_lower"] * (
        b["t2_lower"] + b["m4_lower"] * b["D_lower"]
    ) / b["L_upper"]
    if Decimal(cert["theorem_decimal"]) > assembled:
        raise ValueError("advertised lower decimal exceeds conservative assembly")
    if Decimal(cert["theorem_decimal"]) > ball_lower(cert["kappa_lower"]):
        raise ValueError("advertised lower decimal exceeds Arb result")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("certificate", type=Path)
    args = parser.parse_args()
    verify(args.certificate)
    print("lower certificate wrapper: ACCEPT")
