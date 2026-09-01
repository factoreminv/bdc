"""Fixed transition matrices of the filter model.

`filter_window.WinModel` builds the two channel matrices `T_0`, `T_1` from this
function, so the verification path imports it.  In the working tree this function
lived in the module that also held the cutting-plane solver which *searched* for the
test measure and the potential.  That solver is a search procedure: it proposed the
fixed objects shipped in `upper/results/`, and nothing in the proof depends on it, so
only the transition-matrix construction is reproduced here.  See the Fixed-Object
Verification Principle in the paper and the corresponding note in the top-level
`README.md`.

`smat` below is reproduced character-for-character from `src/filter_cert.py` in the
author's development tree.  It is a pure function of `(M, c, d)` with no state, no
search, and no dependence on anything else in that tree.  That this is the revision
which produced the archived run is for the author to confirm; a full replay that
reproduces the recorded endpoint would corroborate it.
"""
import numpy as np


def smat(M, c, d):
    """T_c = d I + p S_c  as a 2^M x 2^M matrix."""
    n = 1 << M
    p = 1.0 - d
    A = d * np.eye(n)
    half = 1 << (M - 1)
    for v in range(n):
        if (v & 1) == c:
            A[v, v >> 1] += p
            A[v, (v >> 1) + half] += p
    return A
