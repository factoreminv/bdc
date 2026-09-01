"""RIGOROUS upper bound on the filter-window reward, by directed-rounding interval arithmetic.

The certificate is  C(d) <= theta = max_{s,a} [ rbar(s,a) + h(s') - h(s) ]  for ANY h.  Two
observations make a fully rigorous theta cheap:

  * h is arbitrary.  Its stored float values ARE the h we use, so they are exact rationals and
    contribute NO error.  Only rbar needs to be bounded from above, and the final max must be
    rounded up.
  * The window recursion is entirely non-negative, so it has no cancellation; the only delicate
    steps are the entropy difference (a Jensen-Shannon divergence, where cancellation is real)
    and the gradient.

Soundness of the charge needs one extra care that the float code did not need: concavity
licenses the TRUE gradient as a supergradient, and a rounded gradient is not one.  So g is
carried as an ENCLOSURE [g_lo, g_hi] and each term of the charge picks the direction that
maximises it:  mu_hat - C >= 0, so -<g, mu_hat - C> is largest at g_lo; W_j >= 0, so the
Gmax_j terms take their max over g_hi.

d is NOT representable in binary, so the recursion is seeded with an interval that contains
the REAL number (e.g. the real 0.65, not float64(0.65)), and pi_m is computed as an EXACT
RATIONAL from d = 13/20 and rounded up.
"""
import numpy as np
from fractions import Fraction
from math import comb
from src import ivl
from src.filter_win2 import _wslice


def pi_exact_up(m, M, d_frac):
    """pi_m = P(Bin(m, p) < M) as an EXACT rational, rounded UP to a float64."""
    p = 1 - d_frac
    tot = sum(Fraction(comb(m, j)) * p ** j * d_frac ** (m - j) for j in range(M))
    x = np.float64(float(tot))
    while Fraction(float(x)) < tot:            # ensure the float is an upper bound
        x = np.nextafter(x, np.inf)
    return x, tot


class IvModel:
    """interval mirror of WinModel2."""

    def __init__(self, d_frac, L, k, m):
        self.L, self.k, self.m = L, k, m
        self.M = M = max(L, k)
        self.d_frac = d_frac
        dc = np.float64(float(d_frac))
        self.d = (np.nextafter(dc, -np.inf), np.nextafter(dc, np.inf))
        self.p = ivl.sub(ivl.const(1.0), self.d)
        self.pi_up, self.pi_frac = pi_exact_up(m, M, d_frac)
        self.nW = (1 << M) - 1

    # ---- recursion -------------------------------------------------------
    def start(self):
        C = (np.zeros((1, 1 << self.M)), np.zeros((1, 1 << self.M)))
        Wl = np.zeros((1, self.nW)); Wh = np.zeros((1, self.nW))
        Wl[0, 0] = 1.0; Wh[0, 0] = 1.0
        return C, (Wl, Wh)

    def step(self, C, W, a):
        M, half = self.M, 1 << (self.M - 1)
        d, p = self.d, self.p
        Cn = ivl.smul_nn(d, C)
        left = ivl.add((C[0][:, :half], C[1][:, :half]), (C[0][:, half:], C[1][:, half:]))
        addn = ivl.smul_nn(p, ivl.add(left, (W[0][:, _wslice(M - 1)], W[1][:, _wslice(M - 1)])))
        Cn0, Cn1 = Cn[0].copy(), Cn[1].copy()
        Cn0[:, a::2] = np.nextafter(Cn0[:, a::2] + addn[0], -np.inf)
        Cn1[:, a::2] = np.nextafter(Cn1[:, a::2] + addn[1],  np.inf)
        Wn = ivl.smul_nn(d, W)
        Wn0, Wn1 = Wn[0].copy(), Wn[1].copy()
        for j in range(M - 1, 0, -1):
            src = (W[0][:, _wslice(j - 1)], W[1][:, _wslice(j - 1)])
            t = ivl.smul_nn(p, src)
            sl = _wslice(j)
            Wn0[:, sl][:, a::2] = np.nextafter(Wn0[:, sl][:, a::2] + t[0], -np.inf)
            Wn1[:, sl][:, a::2] = np.nextafter(Wn1[:, sl][:, a::2] + t[1],  np.inf)
        return (Cn0, Cn1), (Wn0, Wn1)

    def muhat(self, C, W):
        M = self.M
        mu = (C[0].copy(), C[1].copy())
        for j in range(M):
            s = 2.0 ** -(M - j)                       # exact power of two
            t = (np.tile(W[0][:, _wslice(j)], 1 << (M - j)) * s,
                 np.tile(W[1][:, _wslice(j)], 1 << (M - j)) * s)
            mu = ivl.add(mu, t)
        return mu

    # ---- reward ----------------------------------------------------------
    def _marg(self, A, j):
        M = self.M
        if j == M: return A
        n = 1 << (M - j)
        return ivl.rsum((A[0].reshape(A[0].shape[0], n, 1 << j),
                         A[1].reshape(A[1].shape[0], n, 1 << j)), 1, n)

    def _applyT(self, A, c):
        M, half = self.M, 1 << (self.M - 1)
        d, p = self.d, self.p
        B = ivl.smul_nn(d, A)
        s = ivl.smul_nn(p, ivl.add((A[0][:, :half], A[1][:, :half]),
                                   (A[0][:, half:], A[1][:, half:])))
        B0, B1 = B[0].copy(), B[1].copy()
        B0[:, c::2] = np.nextafter(B0[:, c::2] + s[0], -np.inf)
        B1[:, c::2] = np.nextafter(B1[:, c::2] + s[1],  np.inf)
        return (B0, B1)

    def _applyTt(self, Y, c):
        M, half = self.M, 1 << (self.M - 1)
        d, p = self.d, self.p
        B = ivl.smul(d, Y)
        yc = ivl.smul(p, (Y[0][:, c::2], Y[1][:, c::2]))
        B0, B1 = B[0].copy(), B[1].copy()
        B0[:, :half] = np.nextafter(B0[:, :half] + yc[0], -np.inf)
        B1[:, :half] = np.nextafter(B1[:, :half] + yc[1],  np.inf)
        B0[:, half:] = np.nextafter(B0[:, half:] + yc[0], -np.inf)
        B1[:, half:] = np.nextafter(B1[:, half:] + yc[1],  np.inf)
        return (B0, B1)

    @staticmethod
    def _expand(x, M, j):
        if j == M: return x
        return (np.tile(x[0], (1, 1 << (M - j))), np.tile(x[1], (1, 1 << (M - j))))

    def rbar_ub(self, C, W, c, w):
        """UPPER bound on  r(muhat,c) + charge, valid for every mu in Omega(u)."""
        M, L, k = self.M, self.L, self.k
        d, p = self.d, self.p
        A = self.muhat(C, W)
        Tc = self._applyT(A, c)
        AL, ALm, TL = self._marg(A, L), self._marg(A, L - 1), self._marg(Tc, L)
        # js is SUBTRACTED, so we need a LOWER bound on it
        HTL = ivl.entropy(TL, 1 << L)
        HAL = ivl.entropy(AL, 1 << L)
        HALm = ivl.entropy(ALm, 1 << (L - 1))
        js = ivl.sub(HTL, ivl.add(ivl.smul(p, HALm), ivl.smul(d, HAL)))
        wc = np.ascontiguousarray(w[c], np.float64)
        Ak = self._marg(A, k)
        desc = ivl.smul(p, ivl.dot_signed(Ak, wc[None, :], 1 << k))
        r = ivl.sub(desc, js)                                   # r[1] is the upper bound
        # gradient ENCLOSURE
        gw = np.tile(wc, 1 << (M - k))[None, :]
        gterm = ivl.sub(self._applyTt(self._expand(ivl.glog(TL), M, L), c),
                        ivl.add(ivl.smul(p, self._expand(ivl.glog(ALm), M, L - 1)),
                                ivl.smul(d, self._expand(ivl.glog(AL), M, L))))
        g = ivl.sub(ivl.smul(p, (gw, gw)), gterm)
        # charge, each term in the direction that maximises it
        # EXACT joint-vertex charge (see filter_win2.reward), in interval form:
        #   max_{mu in Omega} <g,mu> = max_x [ <g,C> + sum_j V_j[x mod 2^{M-j}] ],
        # upper-bounded termwise, minus a LOWER bound on <g, muhat>.
        n = 1 << M
        gam = (n * ivl.U) / (1 - n * ivl.U)
        nb = g[1].shape[0]
        cC = np.where(g[1] >= 0, C[1], C[0])                    # C >= 0
        t0 = (g[1] * cC).sum(axis=1)
        t0mag = (np.abs(g[1]) * np.maximum(np.abs(C[0]), np.abs(C[1]))).sum(axis=1)
        tot = np.repeat(np.nextafter(t0 + gam * t0mag, np.inf)[:, None], n, axis=1)
        for j in range(M):
            gj = g[1].reshape(nb, 1 << (M - j), 1 << j)
            Wlo = W[0][:, _wslice(j)]; Whi = W[1][:, _wslice(j)]
            # The maximizing endpoint depends on the sign of EACH coefficient
            # g[h,s].  Choosing once per s after maxing over h is unsound when a
            # column contains both signs: Whi then makes a negative product too
            # small.  Broadcast the interval endpoints over h and select
            # coefficientwise.
            Wsel = np.where(gj >= 0, Whi[:, None, :], Wlo[:, None, :])
            Vj = np.einsum('ihs,ihs->ih', gj, Wsel)
            Vjm = np.einsum('ihs,is->ih', np.abs(gj), np.maximum(np.abs(Wlo), np.abs(Whi)))
            idx = np.arange(n) & ((1 << (M - j)) - 1)
            tot = np.nextafter(tot + np.nextafter(Vj + gam * Vjm, np.inf)[:, idx], np.inf)
        cA = np.where(g[0] >= 0, A[0], A[1])                    # A >= 0, LOWER bound
        s0 = (g[0] * cA).sum(axis=1)
        s0mag = (np.abs(g[0]) * np.maximum(np.abs(A[0]), np.abs(A[1]))).sum(axis=1)
        chg = np.nextafter(tot.max(axis=1) - np.nextafter(s0 - gam * s0mag, -np.inf), np.inf)
        old = np.nextafter(self.pi_up * np.nextafter(g[1].max(axis=1) - g[0].min(axis=1),
                                                     np.inf), np.inf)
        return np.nextafter(r[1] + np.minimum(chg, old), np.inf)
