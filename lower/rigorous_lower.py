"""Arb/FLINT proof arithmetic for the PRC lower-bound certificate.

This module deliberately shares no FFT, clipping, or renormalisation code with the
exploratory evaluators.  Positive masses are rounded down to fixed-point integers and
convolved as exact ``fmpz_poly`` objects.  Transcendental functions and final sums use Arb
balls.  Terms whose bin cannot be decided from an Arb enclosure are dropped.
"""
from __future__ import annotations

import math
import numpy as np
from flint import arb, arb_poly, ctx, fmpz, fmpz_poly

ctx.prec = 192
LN2 = arb(2).log()


def _a(x) -> arb:
    """Enclose the exact stored binary64 datum, not a shortened decimal."""
    x = float(x)
    n, d = x.as_integer_ratio()
    return arb(n) / d


def _floor_fmpz(x: arb) -> fmpz:
    z = x.lower().floor().unique_fmpz()
    if z is None:
        raise ArithmeticError(f"could not determine floor of {x}")
    return z


def _certain_bin(x: arb, origin: arb, step: arb, nbins: int) -> int | None:
    """Return the common bin of the entire ball, or None (which means: drop it)."""
    q0 = _floor_fmpz((x.lower() - origin) / step)
    q1 = _floor_fmpz((x.upper() - origin) / step)
    if q0 != q1:
        return None
    q = int(q0)
    return q if 0 <= q < nbins else None


def _g4(s: arb) -> arb:
    """(1+exp(s))*h(sigmoid(s)), in bits; this function is increasing."""
    e = s.exp()
    return ((1 + e) * (1 + e).log() - e * s) / LN2


def correction_m4_arb(a, P, Kmax=400, ds="0.002", lo="-200", hi="200", qbits=176):
    """Rigorous lower bound for the four-run segmentation-type contribution.

    The range restriction and uncertain-bin rejection only delete nonnegative summands.
    Histogram masses are floor(Q*f), so exact integer convolution is coefficientwise no
    larger than the true convolution.  Since g4 is nonnegative and increasing, its value at
    the lower endpoint of the two-bin Minkowski cell is a valid kernel lower bound.  The
    binomial sums in the type weights sum out within-type count allocations; this is a
    contribution to H(T|W,Z), not to the complete-state entropy.
    """
    aa = [_a(x) for x in np.asarray(a, float)]
    pp0 = [_a(x) for x in np.asarray(P, float)]
    ps = sum(pp0, arb(0)); pp = [x / ps for x in pp0]
    step, origin, top = arb(ds), arb(lo), arb(hi)
    nb = int(math.ceil((float(hi) - float(lo)) / float(ds))) + 2
    Q = fmpz(1) << qbits
    H1 = [fmpz(0)] * nb
    H2 = [fmpz(0)] * nb
    fact = fmpz(1)
    for k in range(1, Kmax + 1):
        fact *= k
        for i, z1 in enumerate(aa):
            z1k = z1 ** k
            for j, z3 in enumerate(aa):
                S = z1 + z3
                B = S ** k - z3 ** k
                if not (B > 0):
                    continue
                common = pp[i] * pp[j] * (-S).exp() / fact
                f1 = common * B
                u = (z1k / B).log()
                b = _certain_bin(u, origin, step, nb)
                if b is not None:
                    H1[b] += _floor_fmpz(f1 * Q)

                # side 2 has f=P2 P4 exp(-S) z4^k/k! and
                # v=log(((z2+z4)^k-z4^k)/z4^k).
                z4k = z3 ** k
                f2 = common * z4k
                v = (B / z4k).log()
                b = _certain_bin(v, origin, step, nb)
                if b is not None:
                    H2[b] += _floor_fmpz(f2 * Q)

    C = fmpz_poly(H1) * fmpz_poly(H2)
    total = arb(0)
    q2 = arb(Q) * arb(Q)
    # A value from bins i,j lies in [2*origin+(i+j)step,
    # 2*origin+(i+j+2)step].  g4 is increasing.
    base = 2 * origin
    for r, mass in enumerate(C.coeffs()):
        if mass:
            gl = _g4(base + r * step).lower()
            if gl > 0:
                total += arb(mass) * gl / q2
    return total


def _entropy_term(p: arb) -> arb:
    if not (p > 0):
        return arb(0)
    return -p * p.log() / LN2


def _binary_entropy(p: arb) -> arb:
    return _entropy_term(p) + _entropy_term(1 - p)


def _conditional_count_entropy(a0: arb, s: arb, Kmax: int) -> arb:
    """Upper enclosure for H(Pois(a0)|positive + Pois(s))."""
    ea = (-a0).exp()
    den = 1 - ea
    eas = (-(a0 + s)).exp()
    es = (-s).exp()
    pa = arb(1)
    ps = arb(1)
    total_p = arb(0)
    ent = arb(0)
    for k in range(1, Kmax + 1):
        pa *= (a0 + s) / k
        ps *= s / k
        q = (eas * pa - ea * es * ps) / den
        if q > 0:
            total_p += q
            ent += _entropy_term(q)
    # Entropy of the unresolved tail.  For mass t on positive integer offsets
    # with mean at most mu, the geometric distribution maximises entropy:
    # H_tail <= -t log t + t log(e*(mu/t+1)).
    t = (1 - total_p).upper()
    if t > 0:
        mu = a0 / den + s
        ent += (-t * t.log() + t * (1 + mu / t).log() + t) / LN2
    return ent.upper()


def rdm_arb(a, P, Kmax=500, Rmax=12):
    """One-sided Arb enclosure of the Drinea--Mitzenmacher baseline.

    H(K) is bounded below by retained nonnegative entropy terms.  H(K|T) is evaluated
    exactly for zero and one silent even run; the remaining geometric tail has probability
    D^2 and is bounded by the sharp variance entropy bound for integer-valued variables.
    No truncated distribution is renormalised.
    """
    aa = [_a(x) for x in np.asarray(a, float)]
    pp0 = [_a(x) for x in np.asarray(P, float)]
    psum = sum(pp0, arb(0)); pp = [x / psum for x in pp0]
    D = sum((p * (-z).exp() for p, z in zip(pp, aa)), arb(0))
    L = sum((p * z for p, z in zip(pp, aa)), arb(0))
    nu = (1 - D) / (1 + D)

    # Marginal count law of one free run, retained on 0..Kmax.
    G = [arb(0)] * (Kmax + 1)
    for p, z in zip(pp, aa):
        pk = (-z).exp()
        G[0] += p * pk
        for k in range(1, Kmax + 1):
            pk *= z / k
            G[k] += p * pk
    F = [arb(0)] + [G[k] / (1 - D) for k in range(1, Kmax + 1)]
    gp, fp = arb_poly(G), arb_poly(F)
    cur = fp
    law = arb_poly([arb(0)] * (Kmax + 1))
    for r in range(Rmax + 1):
        law += cur * ((1 - D) * D ** r)
        cur = (cur * gp).truncate(Kmax + 1)
    law = law.truncate(Kmax + 1)
    HK = sum((_entropy_term(x).lower() for x in law.coeffs()), arb(0))

    # Conditional entropy for R=0 and R=1.  Pplus is the law of the emitting first run.
    Pplus = [p * (1 - (-z).exp()) / (1 - D) for p, z in zip(pp, aa)]
    H0 = arb(0); H1 = arb(0)
    for p0, z0 in zip(Pplus, aa):
        if not (p0 > 0):
            continue
        H0 += p0 * _conditional_count_entropy(z0, arb(0), Kmax)
        for p1, z1 in zip(pp, aa):
            H1 += p0 * p1 * _conditional_count_entropy(z0, z1, Kmax)

    # R>=2.  Conditional on this event R=2+Geom_0(1-D).  Average entropy is at most
    # the entropy bound formed from the unconditional variance on this tail.
    EA2 = sum((p * z * z for p, z in zip(pp, aa)), arb(0))
    varA = EA2 - L * L
    EK0 = L / (1 - D)
    EK02 = (EA2 + L) / (1 - D)
    varK0 = EK02 - EK0 * EK0
    ER = 2 + D / (1 - D)
    varR = D / (1 - D) ** 2
    ES = ER * L
    varS = ER * varA + varR * L * L
    varK = varK0 + ES + varS
    Htail = ((2 * arb.pi() * arb(1).exp() * (varK + arb(1) / 12)).log() / (2 * LN2)).upper()
    HKT = ((1 - D) * H0 + (1 - D) * D * H1 + D * D * Htail).upper()

    pen = sum((p * _binary_entropy((-z).exp()) for p, z in zip(pp, aa)), arb(0)) / (1 + D)
    rate = (nu * (HK - HKT) - pen.upper()) / L
    return rate, dict(HK=HK, HKT=HKT, penalty=pen, D=D, L=L, nu=nu,
                      H0=H0, H1=H1, Htail=Htail)


def _kernel_a(x: arb, w: arb) -> arb:
    return (1 + x.exp() + (x + w).exp()).log() / LN2


def _kernel_b(x: arb, y: arb) -> arb:
    return (1 + x.exp() / (1 + y.exp())).log() / LN2


def _kernel_c(u: arb, y: arb) -> arb:
    return (1 + (u + y).exp() / (1 + y.exp())).log() / LN2


def correction_t2_arb(a, P, Kmax=60, ds="0.5", lo="-80", hi="80", qbits=200):
    """Rigorous coarse depth-two type correction using exact 2-D integer convolution.

    The three kernels have simple coordinate monotonicities: ``a`` increases in both
    coordinates, ``b`` increases in x and decreases in y, and ``c`` increases in both u
    and y.  Their cell minima are therefore evaluated at proved corners, not sampled.  The
    type weights have already summed out the compatible within-type count allocations, so
    the result is a retained part of H(T|W,Z).
    """
    aa = [_a(x) for x in np.asarray(a, float)]
    pp0 = [_a(x) for x in np.asarray(P, float)]
    ps = sum(pp0, arb(0)); pp = [x / ps for x in pp0]
    step, origin = arb(ds), arb(lo)
    nb = int(math.ceil((float(hi) - float(lo)) / float(ds))) + 2
    outn = 2 * nb - 1
    stride = outn
    Q = fmpz(1) << qbits
    q3 = arb(Q) ** 3

    # Middle factors do not depend on z3.
    mids = {q: [fmpz(0)] * (nb * stride) for q in "abc"}
    fact = fmpz(1)
    for k in range(1, Kmax + 1):
        fact *= k
        for i, z2 in enumerate(aa):
            z2k = z2 ** k
            for j, z4 in enumerate(aa):
                z4k = z4 ** k
                B = (z2 + z4) ** k - z4k
                if not (B > 0):
                    continue
                beta = (z4k / B).log(); gamma = (z2k / B).log()
                common = pp[i] * pp[j] * (-(z2 + z4)).exp() / fact
                data = {
                    "a": (-beta, gamma, common * z4k),
                    "b": ( beta, gamma, common * B),
                    "c": ( beta, -gamma, common * z2k),
                }
                for q, (x, y, mass) in data.items():
                    ix = _certain_bin(x, origin, step, nb)
                    iy = _certain_bin(y, origin, step, nb)
                    if ix is not None and iy is not None:
                        mids[q][ix + stride * iy] += _floor_fmpz(mass * Q)
    midpoly = {q: fmpz_poly(v) for q, v in mids.items()}

    total = arb(0)
    for i3, z3 in enumerate(aa):
        left = {q: [fmpz(0)] * nb for q in "abc"}
        right = {q: [fmpz(0)] * nb for q in "abc"}
        fact = fmpz(1)
        outer = pp[i3] * (-z3).exp()
        for k in range(1, Kmax + 1):
            fact *= k
            for i, z1 in enumerate(aa):
                z1k = z1 ** k
                B = (z1 + z3) ** k - z3 ** k
                if B > 0:
                    alpha = (B / z1k).log()
                    common = outer * pp[i] * (-z1).exp() / fact
                    for q, x, mass in (("a", -alpha, common * B),
                                       ("b", alpha, common * z1k),
                                       ("c", alpha, common * z1k)):
                        ix = _certain_bin(x, origin, step, nb)
                        if ix is not None:
                            left[q][ix] += _floor_fmpz(mass * Q)
            for i, z5 in enumerate(aa):
                z5k = z5 ** k
                B = (z3 + z5) ** k - z5k
                if B > 0:
                    delta = (B / z5k).log()
                    common = pp[i] * (-z5).exp() / fact
                    for q, y, mass in (("a", delta, common * z5k),
                                       ("b", delta, common * z5k),
                                       ("c", -delta, common * B)):
                        iy = _certain_bin(y, origin, step, nb)
                        if iy is not None:
                            right[q][iy] += _floor_fmpz(mass * Q)

        for q in "abc":
            # HL uses x exponents; HR uses stride*y exponents.
            rcoeff = [fmpz(0)] * (stride * (nb - 1) + 1)
            for i, c in enumerate(right[q]):
                if c:
                    rcoeff[stride * i] = c
            rp = fmpz_poly(rcoeff)
            C = fmpz_poly(left[q]) * midpoly[q] * rp
            base = 2 * origin
            subtotal = arb(0)
            for exponent, mass in enumerate(C.coeffs()):
                if not mass:
                    continue
                ix, iy = exponent % stride, exponent // stride
                xlo = base + ix * step
                ylo = base + iy * step
                # Each coordinate is the sum of one middle and one side bin.
                if q == "a":
                    g = _kernel_a(xlo, ylo)
                elif q == "b":
                    g = _kernel_b(xlo, ylo + 2 * step)
                else:
                    g = _kernel_c(xlo, ylo)
                gl = g.lower()
                if gl > 0:
                    subtotal += arb(mass) * gl / q3
            total += subtotal.lower()
    return total


if __name__ == "__main__":
    A = np.load("out/P_CERT_BEST.npy")
    law, prob = A[0], A[1] / A[1].sum()
    v = correction_m4_arb(law, prob)
    print("m4 Arb enclosure:", v)
    print("m4 certified lower endpoint:", v.lower())
    rate, info = rdm_arb(law, prob)
    print("R_DM Arb lower enclosure:", rate)
    print("R_DM certified lower endpoint:", rate.lower())
    print("R_DM components:", info)
