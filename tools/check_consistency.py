"""Cross-check the README, the manifests, the Lean source, and the two constants.

Every claim this supplement makes about itself is stated in more than one place: the
paper's constants, the upper manifest, the lower certificate, the Lean decimals, and
the README prose.  This script fails if any of them drift apart.  Run from the
repository root:

    python tools/check_consistency.py

It reads only committed files, needs no dependencies beyond the standard library, and
does not run either calculation.
"""
from __future__ import annotations

import json
import re
import sys
from decimal import Decimal
from pathlib import Path

UPPER_DECIMAL = "0.250984"
LOWER_DECIMAL = "0.12415"

LEAN_SCOPE_PHRASES = [
    "Bellman telescoping implication",
    "geometric tail inequality",
    "abstract upper-bound assembly lemma",
    "monotonicity-extension algebra",
    "retained nonnegative corrections",
    "lower-bound decimal arithmetic",
    "not an end-to-end formalization of either capacity bound",
]
LEAN_THEOREMS = {
    "retained_corrections_lower", "prc_lower_assembly", "lower_decimal_arithmetic",
    "headline_lower_012415", "cert_bound", "cert_rate", "geom_sum_mul",
    "geom_sum_le_inv", "geom_tail", "assembly", "extend_in_d",
}
ARITHMETIC_PHRASES = [
    "NumPy binary64 interval endpoints", "300-bit", "four ulps",
    "diagnostic, not a proof", "round-to-nearest", "fast-math", "flush-to-zero",
    "reassociation", "search procedures",
    "does not assume that either search converged",
    "does not repeat the `2^30`-state Bellman calculation",
]

failures: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not cond:
        failures.append(label)


def ball_lower(text: str) -> Decimal:
    return Decimal(text.strip()[1:-1].split("+/-")[0].strip())


def main() -> int:
    readme = " ".join(Path("README.md").read_text().split())   # whitespace-insensitive
    cert = Path("lean/Bdcproof/Cert.lean").read_text()
    up = json.loads(Path("upper/results/rigor_d0.65_L6_k6_m30_manifest.json").read_text())
    lo = json.loads(Path("lower/out/rigorous_lower_certificate.json").read_text())

    print(f"upper bound {UPPER_DECIMAL}")
    check("manifest endpoint <= theorem decimal",
          up["theta_hi_over_one_minus_d"] <= float(up["theorem_decimal"]),
          f'{up["theta_hi_over_one_minus_d"]} <= {up["theorem_decimal"]}')
    check("manifest decimal is the paper's constant", up["theorem_decimal"] == UPPER_DECIMAL)
    check("README quotes it", f"checks the coefficient `{UPPER_DECIMAL}`" in readme)
    check("exhaustive over 2^30 states", up["states_checked"] == 2 ** 30)
    check("bias is the 2^28-state potential", up["bias_state_bits"] == 28)

    print(f"\nlower bound {LOWER_DECIMAL}")
    check("certificate decimal is the paper's constant", lo["theorem_decimal"] == LOWER_DECIMAL)
    check("Arb result >= theorem decimal",
          ball_lower(lo["kappa_lower"]) >= Decimal(LOWER_DECIMAL),
          str(ball_lower(lo["kappa_lower"]))[:24])
    check("README quotes it", f"checks the coefficient `{LOWER_DECIMAL}`" in readme)

    b = {k: Decimal(v) for k, v in lo["lean_component_bounds"].items()}
    for k in ("rdm_lower", "m4_lower", "t2_lower", "D_lower", "nu_lower"):
        check(f"{k} decimal weakens the Arb ball", b[k] <= ball_lower(lo[k]))
    check("L_upper decimal weakens the Arb ball", b["L_upper"] >= ball_lower(lo["L_upper"]))
    assembled = b["rdm_lower"] + b["nu_lower"] * (
        b["t2_lower"] + b["m4_lower"] * b["D_lower"]) / b["L_upper"]
    check("weakened decimals still assemble to the constant",
          assembled >= Decimal(LOWER_DECIMAL), str(assembled)[:24])

    print("\nLean")
    m = re.search(
        r"lower_decimal_arithmetic\s*:\s*\(([\d.]+) : ℝ\) ≤ ([\d.]+) \+ ([\d.]+) \*\s*"
        r"\(([\d.]+) \+ ([\d.]+) \* ([\d.]+)\) / ([\d.]+)", cert, re.S)
    check("lower_decimal_arithmetic is present and parses", m is not None)
    if m:
        dec, rdm, nu, t2, m4, D, L = (Decimal(x) for x in m.groups())
        check("Lean decimal is the paper's constant", dec == Decimal(LOWER_DECIMAL))
        for name, v in (("rdm_lower", rdm), ("nu_lower", nu), ("t2_lower", t2),
                        ("m4_lower", m4), ("D_lower", D), ("L_upper", L)):
            check(f"Lean {name} matches the certificate", v == b[name], str(v))
    check("declarations are exactly the documented set",
          set(re.findall(r"^theorem (\w+)", cert, re.M)) == LEAN_THEOREMS)
    check("no obsolete headline constants",
          not re.search(r"0\.31330422|0\.15665211", cert))
    check("no sorry or admit in the Lean sources",
          not any(re.search(r"\b(sorry|admit)\b", p.read_text())
                  for p in Path("lean").rglob("*.lean")))
    for phrase in LEAN_SCOPE_PHRASES:
        check(f"README states: {phrase}", phrase in readme)

    print("\narithmetic and scope")
    for phrase in ARITHMETIC_PHRASES:
        check(f"README states: {phrase}", phrase in readme)

    print()
    if failures:
        print(f"INCONSISTENT: {len(failures)} check(s) failed")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("CONSISTENT: README, manifests, Lean, and the paper's constants agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
