"""Recompute and store the outward-rounded lower-bound certificate."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from rigorous_lower import correction_m4_arb, correction_t2_arb, rdm_arb


def lower_string(x):
    return str(x.lower())


if __name__ == "__main__":
    source = Path("out/P_CERT_BEST.npy")
    A = np.load(source)
    a, P = A[0], A[1] / A[1].sum()
    started = time.time()
    rdm, info = rdm_arb(a, P, Kmax=500, Rmax=12)
    m4 = correction_m4_arb(a, P, Kmax=400, ds="0.002", lo="-200", hi="200", qbits=176)
    t2 = correction_t2_arb(a, P, Kmax=60, ds="0.5", lo="-80", hi="80", qbits=300)
    # The m=6 term is nonnegative and is omitted.  All retained factors are positive.
    final = rdm.lower() + info["nu"].lower() * (
        t2.lower() + m4.lower() * info["D"].lower()
    ) / info["L"].upper()
    cert = {
        "law_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "law_file": source.name,
        "precision_bits": 192,
        "parameters": {
            "rdm_Kmax": 500, "rdm_Rmax": 12,
            "m4_Kmax": 400, "m4_step": "0.002",
            "m4_window": ["-200", "200"], "m4_fixed_point_bits": 176,
            "t2_Kmax": 60, "t2_step": "0.5",
            "t2_window": ["-80", "80"], "t2_fixed_point_bits": 300,
        },
        "arithmetic": "python-flint Arb 192-bit balls; exact fmpz polynomial convolution",
        "rdm_lower": lower_string(rdm),
        "m4_lower": lower_string(m4),
        "t2_lower": lower_string(t2),
        "D_lower": lower_string(info["D"]),
        "nu_lower": lower_string(info["nu"]),
        "L_upper": str(info["L"].upper()),
        "m6_retained": False,
        "kappa_lower": lower_string(final),
        "theorem_decimal": "0.12415",
        "lean_component_bounds": {
            "rdm_lower": "0.119876",
            "m4_lower": "0.037617",
            "t2_lower": "0.030455",
            "D_lower": "0.058937",
            "nu_lower": "0.888686",
            "L_upper": "6.780381"
        },
        "elapsed_seconds": time.time() - started,
    }
    out = Path("out/rigorous_lower_certificate.json")
    out.write_text(json.dumps(cert, indent=2) + "\n")
    print(json.dumps(cert, indent=2))
