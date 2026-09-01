"""TIGHTER polytope charge for the filter-window certificate.

Theorem G bounds  max_{mu in Omega(u)} r(mu,c)  by  r(muhat,c) + pi_m (max_j g_j - min_j g_j)
with g = grad r(muhat,c).  That uses only  ||mu - muhat||_1 <= 2 pi_m.  The residual of
A_u is far more structured than an l1 ball: decomposing A_u by the number j < M of survivors
inside the window,

    A_u nu = C_u  +  sum_{j<M} sum_{s in {0,1}^j} W_j[s] * ( prepend s to nu|_{M-j} ),

where C_u is the (constant) >= M-survivor part and  sum_j sum_s W_j[s] = pi_m.  Hence

    max_{nu} <g, A_u nu - muhat>  <=  sum_j <W_j, Gmax_j>  -  <g, muhat - C_u>,          (T)
    Gmax_j[s] := max over the M-j free bits of g[ s . * ],

both terms EXACT/computable from the same recursion.  (T) <= pi_m (max g - min g) always,
and is much smaller in practice because W is concentrated on j = M-1, where Gmax_{M-1} is a
max over ONE free bit rather than over everything.

Recursion, prepending symbol a (i.e. applying T_a on the outside):
    C     <- d C + p S_a C + p promote_a(W_{M-1});     W_j <- d W_j + p prepend_a(W_{j-1})
with prepend_a(index t) = 2t + a and S_a as usual.  Start: W_0 = [1], C = 0.
"""
import numpy as np, time
from scipy.stats import binom
from src.filter_window import _marg, _expand, _Hrow, _glog, _apply_T, _apply_Tt

# packed layout for W: block j occupies [2^j - 1, 2^{j+1} - 1)
def _wslice(j):
    return slice((1 << j) - 1, (1 << (j + 1)) - 1)


class WinModel2:
    def __init__(self, d, L, k, m):
        self.d, self.p, self.L, self.k, self.m = d, 1 - d, L, k, m
        self.M = M = max(L, k)
        self.pi = float(binom.cdf(M - 1, m, 1 - d))
        self.nW = (1 << M) - 1

    def start(self, dtype=np.float32):
        C = np.zeros((1, 1 << self.M), dtype)
        W = np.zeros((1, self.nW), dtype); W[0, 0] = 1.0
        return C, W

    def step(self, C, W, a):
        """apply T_a on the outside."""
        d, p, M = self.d, self.p, self.M
        half = 1 << (M - 1)
        Cn = C * d
        Cn[:, a::2] += p * (C[:, :half] + C[:, half:])
        Cn[:, a::2] += p * W[:, _wslice(M - 1)]          # promotion to >= M survivors
        Wn = W * d
        for j in range(M - 1, 0, -1):
            Wn[:, _wslice(j)][:, a::2] += p * W[:, _wslice(j - 1)]
        return Cn, Wn

    def muhat(self, C, W):
        """A_u applied to the UNIFORM start (a point of Omega(u))."""
        M = self.M
        mu = C.copy()
        for j in range(M):
            mu += np.tile(W[:, _wslice(j)], 1 << (M - j)) * (2.0 ** -(M - j))
        return mu

    def reward(self, C, W, c, w):
        M, L, k, p, d = self.M, self.L, self.k, self.p, self.d
        A = self.muhat(C, W)
        Tc = _apply_T(A, c, d, M)
        AL, ALm, TL = _marg(A, M, L), _marg(A, M, L - 1), _marg(Tc, M, L)
        js = _Hrow(TL) - p * _Hrow(ALm) - d * _Hrow(AL)
        r = p * (_marg(A, M, k) @ w[c].astype(A.dtype)) - js
        g = (p * np.tile(w[c].astype(A.dtype), 1 << (M - k))[None, :]
             - (_apply_Tt(_expand(_glog(TL), M, L), c, d, M)
                - p * _expand(_glog(ALm), M, L - 1)
                - d * _expand(_glog(AL), M, L)))
        # EXACT linearisation maximum over Omega(u) = A_u Delta.
        #
        #   max_{mu in Omega} <g,mu> = max over VERTICES e_x of <g, A_u e_x>,
        #   <g, A_u e_x> = <g,C_u> + sum_j  V_j[ x mod 2^{M-j} ],
        #   V_j[h] = <W_j, g[h<<j : (h<<j)+2^j]>.
        #
        # The previous form took  sum_j <W_j, Gmax_j>  with Gmax_j maximised over the free bits
        # SEPARATELY for each survivor-count block j, i.e. it allowed a different vertex per
        # block.  Maximising jointly over the single x is both exact and the same cost, and is
        # 7-17x tighter in practice (the slack over the true max falls from 0.0236 to 0.0034 at
        # m=20, and 0.00283 to 0.00017 at m=28).  Still an upper bound on the true max over
        # Omega by concavity, so the certificate is unaffected in kind, only in tightness.
        nb = g.shape[0]
        tot = np.einsum('ij,ij->i', g, C)[:, None] + np.zeros((nb, 1 << M), g.dtype)
        for j in range(M):
            Vj = np.einsum('ihs,is->ih',
                           g.reshape(nb, 1 << (M - j), 1 << j), W[:, _wslice(j)])
            idx = np.arange(1 << M) & ((1 << (M - j)) - 1)
            tot += Vj[:, idx]
        chg = tot.max(axis=1) - np.einsum('ij,ij->i', g, A)
        old = self.pi * (g.max(axis=1) - g.min(axis=1))
        return r + np.minimum(chg, old)


def blocks2(md, m1=20, dtype=np.float32):
    M, m = md.M, md.m
    m1 = min(m1, m)
    for P in range(1 << (m - m1)):
        C, W = md.start(dtype)
        for j in range(m - 1, m1 - 1, -1):
            C, W = md.step(C, W, (P >> (j - m1)) & 1)
        for _ in range(m1):
            C0, W0 = md.step(C, W, 0); C1, W1 = md.step(C, W, 1)
            Cn = np.empty((2 * C.shape[0], C.shape[1]), dtype)
            Wn = np.empty((2 * W.shape[0], W.shape[1]), dtype)
            Cn[0::2] = C0; Cn[1::2] = C1; Wn[0::2] = W0; Wn[1::2] = W1
            C, W = Cn, Wn
        yield P << m1, C, W
