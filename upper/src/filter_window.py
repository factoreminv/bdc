"""FILTER REWARD on a WINDOW state: rigorous, and at the repo's proven 2^23-state scale.

The filter state mu_{i+1} (law of the first M output bits of the suffix) is determined by the
next m input symbols up to the polytope

    Omega(u) = A_u Delta,     A_u = T_{u_1} ... T_{u_m},

whose residual (non-constant) part has total mass EXACTLY  pi_m = P(Bin(m,p) < M):  every
survival pattern with at least M survivors makes S_{c_1}...S_{c_M}... a CONSTANT map.  Hence
for mu in Omega(u),  ||mu - muhat(u)||_1 <= 2 pi_m  with muhat(u) = A_u (uniform).

The per-symbol reward r(mu,c) = p<w_c, mu|_k> - JS_p(S_c mu, mu) is CONCAVE in mu, so

    max_{mu in Omega(u)} r(mu,c)  <=  r(muhat,c) + pi_m * ( max_i d_i r - min_i d_i r )    (*)

is rigorous from ONE gradient evaluation -- no Fannes bound, no eps log(1/eps), and the
charge is computed per window rather than worst-cased.  That reward is then fed to exactly
the repo's average-reward MDP + Bellman certificate (Step 9 of PROOF.md).

    C(d) <= theta = max_{s,a} [ rbar(s,a) + h(s') - h(s) ]        for ANY h.

State s = the next m input symbols (bit j of s = x_{i+1+j}); action a = x_i; the shift is
s' = ((s << 1) & (2^m - 1)) | a  because the filter runs BACKWARD along the input.
"""
import numpy as np, time
from scipy.stats import binom
from src.filter_cert import smat

LN2 = np.log(2.0)


def _marg(A, M, j):
    return A if j == M else A.reshape(A.shape[0], 1 << (M - j), 1 << j).sum(axis=1)


def _expand(x, M, j):
    return x if j == M else np.tile(x, (1, 1 << (M - j)))


def _Hrow(V):
    W = np.where(V > 1e-300, V, 1.0)
    return -(V * np.log2(W)).sum(axis=1)


def _glog(V):
    return -np.log2(np.maximum(V, 1e-300)) - 1.0 / LN2


class WinModel:
    def __init__(self, d, L, k, m):
        self.d, self.p, self.L, self.k, self.m = d, 1 - d, L, k, m
        self.M = M = max(L, k)
        self.Mc = [smat(M, c, d) for c in (0, 1)]
        self.pi = float(binom.cdf(M - 1, m, 1 - d))

    def states(self, chunk=1 << 20, dtype=np.float32):
        """generator of (offset, muhat block) over all 2^m states, bit j of s = x_{i+1+j}."""
        M, m = self.M, self.m
        MT = [np.ascontiguousarray(A.T, dtype=dtype) for A in self.Mc]
        A = np.full((1, 1 << M), 2.0 ** -M, dtype=dtype)
        for _ in range(m):                       # A[2t+c] = M_c @ A[t]
            B = np.empty((2 * A.shape[0], 1 << M), dtype=dtype)
            B[0::2] = A @ MT[0]
            B[1::2] = A @ MT[1]
            A = B
        for o in range(0, A.shape[0], chunk):
            yield o, A[o:o + chunk]

    def reward(self, A, c, w, dtype=None):
        """rbar(.,c) of (*) for a block A of filter states (rows).  Sparse T_c / T_c^T."""
        M, L, k, p, d = self.M, self.L, self.k, self.p, self.d
        if dtype is not None:
            A = A.astype(dtype, copy=False)
        Tc = _apply_T(A, c, d, M)
        AL, ALm, TL = _marg(A, M, L), _marg(A, M, L - 1), _marg(Tc, M, L)
        js = _Hrow(TL) - p * _Hrow(ALm) - d * _Hrow(AL)
        r = p * (_marg(A, M, k) @ w[c].astype(A.dtype)) - js
        gr = (p * np.tile(w[c].astype(A.dtype), 1 << (M - k))[None, :]
              - (_apply_Tt(_expand(_glog(TL), M, L), c, d, M)
                 - p * _expand(_glog(ALm), M, L - 1)
                 - d * _expand(_glog(AL), M, L)))
        return r + self.pi * (gr.max(axis=1) - gr.min(axis=1))


def build_reward(md, w, dtype=np.float32, verbose=True):
    n = 1 << md.m
    R = np.empty((n, 2), np.float64)
    t0 = time.time()
    for o, A in md.states(dtype=dtype):
        for c in (0, 1):
            R[o:o + A.shape[0], c] = md.reward(A, c, w)
    if verbose:
        print(f'    reward table: 2^{md.m} states, pi_m = {md.pi:.3e}, '
              f'charge mean {0.0:.0e} [{time.time()-t0:.0f}s]', flush=True)
    return R


def solve(R, m, iters=1500):
    n = 1 << m
    s = np.arange(n, dtype=np.int64)
    nxt = np.empty((n, 2), np.int64)
    for a in (0, 1):
        nxt[:, a] = ((s << 1) & (n - 1)) | a
    h = np.zeros(n)
    for _ in range(iters):
        v = np.maximum(R[:, 0] + h[nxt[:, 0]], R[:, 1] + h[nxt[:, 1]])
        h = v - v[0]
    return h, nxt


def certify(R, h, nxt):
    return float((np.maximum(R[:, 0] + h[nxt[:, 0]], R[:, 1] + h[nxt[:, 1]]) - h).max())


# ---------------------------------------------------------------- chunked build (large m)
def _apply_T(A, c, d, M, out=None):
    """sparse T_c on a row-batch:  (T_c mu)[2w+c] += p (mu[w] + mu[w+half]); rest d*mu."""
    p = 1.0 - d
    half = 1 << (M - 1)
    B = (A * d) if out is None else np.multiply(A, d, out=out)
    B[:, c::2] += p * (A[:, :half] + A[:, half:])
    return B


def _apply_Tt(Y, c, d, M):
    """row-batch action of T_c^T:  (y @ M_c)_u = d y_u + p y_{2u+c}  (both halves)."""
    p = 1.0 - d
    half = 1 << (M - 1)
    B = Y * d
    yc = Y[:, c::2]
    B[:, :half] += p * yc
    B[:, half:] += p * yc
    return B


def blocks(md, m1=22, dtype=np.float32):
    """(offset, muhat block) over all 2^m states without ever materialising 2^m x 2^M."""
    M, m, d = md.M, md.m, md.d
    m1 = min(m1, m)
    for P in range(1 << (m - m1)):
        v = np.full((1, 1 << M), 2.0 ** -M, dtype=dtype)
        for j in range(m - 1, m1 - 1, -1):           # far positions, applied first
            v = _apply_T(v, (P >> (j - m1)) & 1, d, M)
        A = v
        for _ in range(m1):                           # A[2t+c] = T_c A[t]
            B = np.empty((2 * A.shape[0], 1 << M), dtype=dtype)
            B[0::2] = _apply_T(A, 0, d, M)
            B[1::2] = _apply_T(A, 1, d, M)
            A = B
        yield P << m1, A


def build_reward_chunked(md, w, m1=22, dtype=np.float32, verbose=True):
    n = 1 << md.m
    R = np.empty((n, 2), np.float32)
    t0 = time.time()
    for o, A in blocks(md, m1=m1, dtype=dtype):
        for c in (0, 1):
            R[o:o + A.shape[0], c] = md.reward(A, c, w)
        if verbose and (o >> m1) % 16 == 0:
            print(f'      chunk {o>>m1}/{1<<(md.m-m1)}  [{time.time()-t0:.0f}s]', flush=True)
    return R


def solve_big2(R0, R1, m, iters=300, verbose=True):
    """same as solve_big but with contiguous per-action reward arrays and preallocated
    buffers -- what makes 2^32 states fit in memory and run at a sane speed."""
    n = 1 << m
    h = np.zeros(n, np.float32)
    a = np.empty(n, np.float32); b = np.empty(n, np.float32)
    t0 = time.time()
    for it in range(iters):
        a[:n // 2] = h[0::2]; a[n // 2:] = a[:n // 2]
        b[:n // 2] = h[1::2]; b[n // 2:] = b[:n // 2]
        a += R0; b += R1
        np.maximum(a, b, out=a)
        a -= a[0]
        h, a = a, h
        if verbose and it % 50 == 49:
            print(f'      VI {it+1}/{iters}  [{time.time()-t0:.0f}s]', flush=True)
    return h


def certify_big2(R0, R1, h, m):
    n = 1 << m
    a = np.empty(n, np.float32); b = np.empty(n, np.float32)
    a[:n // 2] = h[0::2]; a[n // 2:] = a[:n // 2]
    b[:n // 2] = h[1::2]; b[n // 2:] = b[:n // 2]
    a += R0; b += R1
    np.maximum(a, b, out=a)
    a -= h
    return float(a.max())


def build_reward_split(md, w, m1=22, dtype=np.float32, verbose=True):
    n = 1 << md.m
    R0 = np.empty(n, np.float32); R1 = np.empty(n, np.float32)
    t0 = time.time()
    for o, A in blocks(md, m1=m1, dtype=dtype):
        R0[o:o + A.shape[0]] = md.reward(A, 0, w)
        R1[o:o + A.shape[0]] = md.reward(A, 1, w)
        if verbose and (o >> m1) % 64 == 0:
            print(f'      chunk {o>>m1}/{1<<(md.m-m1)}  [{time.time()-t0:.0f}s]', flush=True)
    return R0, R1


def solve_big(R, m, iters=300, verbose=True):
    """average-reward VI exploiting  nxt[s,a] = ((s<<1)&(n-1))|a  =>  h[nxt] = tile(h[a::2],2)."""
    n = 1 << m
    h = np.zeros(n, np.float32)
    t0 = time.time()
    for it in range(iters):
        he = np.tile(h[0::2], 2); ho = np.tile(h[1::2], 2)
        np.add(R[:, 0], he, out=he); np.add(R[:, 1], ho, out=ho)
        np.maximum(he, ho, out=he)
        h = he - he[0]
        if verbose and it % 50 == 49:
            print(f'      VI {it+1}/{iters}  [{time.time()-t0:.0f}s]', flush=True)
    return h


def certify_big(R, h, m):
    n = 1 << m
    he = np.tile(h[0::2], 2); ho = np.tile(h[1::2], 2)
    np.add(R[:, 0], he, out=he); np.add(R[:, 1], ho, out=ho)
    np.maximum(he, ho, out=he)
    he -= h
    return float(he.max())
