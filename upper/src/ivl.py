"""Directed-rounding interval arithmetic, vectorised, for the filter reward.

An interval is a pair (lo, hi) of float64 arrays with lo <= true <= hi ELEMENTWISE.

Rounding policy
---------------
* elementwise +, *, - : IEEE gives a correctly rounded result (error <= 0.5 ulp), so widening
  the computed value by ONE ulp in each direction is rigorous.
* reductions (sum / dot) : BLAS reorders the partial sums, so per-scalar nextafter cannot
  hook them.  We use the standard a-priori bound instead: for n terms in any order the
  computed sum s^ obeys  |s^ - s| <= gamma_n * sum|x_i|  with gamma_n = n*u/(1 - n*u),
  u = 2^-53.  We widen by that, computed from a separate sum of |x|.
* log2 : the ONE place where a library function is trusted.  numpy/glibc log2 is documented
  faithful (<= 1 ulp); we widen by LOG2_ULP = 4 ulps.  `validate_log2` spot-checks this
  against mpmath at 200 bits on adversarial inputs.
"""
import numpy as np

U = 2.0 ** -53
LOG2_ULP = 4


# Widening.  np.nextafter is the TIGHTEST one-ulp widening but it dominated the profile
# (>60% of the interval pass).  Multiplicative widening is one multiply and is rigorous:
# consecutive float64s differ by at most |x| * 2^-52, and a correctly rounded operation errs
# by at most 0.5 ulp <= |x| * 2^-53, so  x -> x +/- (|x| * 2^-52 + TINY)  covers one ulp with
# margin.  TINY covers zero and the subnormal range.  It is ~2x looser than nextafter per
# operation, which is irrelevant here: the resulting interval width is ~1e-13 against a
# 1e-4 budget.  `validate_log2` and `rigor_validate` re-check the enclosure end to end.
_EPS = 2.0 ** -52
TINY = 1e-300

def _dn(x):   return x - (np.abs(x) * _EPS + TINY)
def _up(x):   return x + (np.abs(x) * _EPS + TINY)
def _dnk(x, k):
    return x - (np.abs(x) * (k * _EPS) + k * TINY)
def _upk(x, k):
    return x + (np.abs(x) * (k * _EPS) + k * TINY)


def const(x):
    """the exact float64 value x as a degenerate interval."""
    return (np.asarray(x, np.float64), np.asarray(x, np.float64))


def around(x, rel=0.0):
    """an interval containing the REAL number that the literal x denotes, widened by rel."""
    a = np.asarray(x, np.float64)
    return (_dn(a * (1 - rel) if rel else a), _up(a * (1 + rel) if rel else a))


def add(a, b):
    return (_dn(a[0] + b[0]), _up(a[1] + b[1]))


def sub(a, b):
    return (_dn(a[0] - b[1]), _up(a[1] - b[0]))


def mul(a, b):
    """general sign-agnostic product."""
    p = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
    lo = np.minimum(np.minimum(p[0], p[1]), np.minimum(p[2], p[3]))
    hi = np.maximum(np.maximum(p[0], p[1]), np.maximum(p[2], p[3]))
    return (_dn(lo), _up(hi))


def mul_nn(a, b):
    """product of two NON-NEGATIVE intervals: monotone, so only two multiplies."""
    return (_dn(a[0] * b[0]), _up(a[1] * b[1]))


def smul_nn(s, a):
    """non-negative scalar interval times a non-negative interval."""
    return (_dn(np.float64(s[0]) * a[0]), _up(np.float64(s[1]) * a[1]))


def smul(s, a):
    """scalar interval s times interval a."""
    return mul((np.float64(s[0]), np.float64(s[1])), a)


def rsum(a, axis, n):
    """sum over `axis` of a NON-NEGATIVE interval, with the gamma_n reduction bound."""
    g = (n * U) / (1 - n * U)
    lo = a[0].sum(axis=axis); hi = a[1].sum(axis=axis)
    return (_dn(lo * (1 - g)), _up(hi * (1 + g)))


def dot_signed(a, bexact, n):
    """<a, b> for an interval a and an EXACT float64 vector b (shape broadcastable).

    Bound: |computed - true| <= gamma_n * sum |a_i b_i|, plus the interval width of a."""
    g = (n * U) / (1 - n * U)
    lo = (np.where(bexact >= 0, a[0], a[1]) * bexact).sum(axis=1)
    hi = (np.where(bexact >= 0, a[1], a[0]) * bexact).sum(axis=1)
    mag = (np.maximum(np.abs(a[0]), np.abs(a[1])) * np.abs(bexact)).sum(axis=1)
    return (_dn(lo - g * mag), _up(hi + g * mag))


def dot_iv(a, b, n):
    """<a, b> for two intervals (both may be signed)."""
    g = (n * U) / (1 - n * U)
    p = mul(a, b)
    lo = p[0].sum(axis=1); hi = p[1].sum(axis=1)
    mag = np.maximum(np.abs(p[0]), np.abs(p[1])).sum(axis=1)
    return (_dn(lo - g * mag), _up(hi + g * mag))


def log2(a, floor=1e-300):
    """log2 of a non-negative interval, clamped below at `floor` (monotone increasing)."""
    lo = np.log2(np.maximum(a[0], floor))
    hi = np.log2(np.maximum(a[1], floor))
    return (_dnk(lo, LOG2_ULP), _upk(hi, LOG2_ULP))


INV_LN2_LO = _dn(np.float64(1.4426950408889634))   # 1/ln 2, both sides
INV_LN2_HI = _up(np.float64(1.4426950408889634))


def negxlog2x(a):
    """elementwise bounds on f(x) = -x log2 x over x in [lo, hi] <= 1.

    f is concave with its maximum at x = 1/e; f(0) = 0."""
    lo_, hi_ = np.maximum(a[0], 0.0), np.maximum(a[1], 0.0)
    # log2 of each endpoint ONCE, then reuse for both the upper and the lower branch
    l_lo = np.log2(np.maximum(lo_, 1e-300))
    l_hi = np.log2(np.maximum(hi_, 1e-300))
    fa_u = _up(-lo_ * _dnk(l_lo, LOG2_ULP))
    fb_u = _up(-hi_ * _dnk(l_hi, LOG2_ULP))
    fa_d = _dn(-lo_ * _upk(l_lo, LOG2_ULP))
    fb_d = _dn(-hi_ * _upk(l_hi, LOG2_ULP))
    INV_E = 0.36787944117144233
    peak = np.float64(_up(INV_E * INV_LN2_HI))          # f(1/e) = 1/(e ln2)
    contains = (lo_ <= INV_E) & (hi_ >= INV_E)
    up = np.where(contains, peak, np.maximum(fa_u, fb_u))
    dn = np.minimum(fa_d, fb_d)
    # f(x) >= 0 on [0,1]; and at x = 0 exactly, f = 0
    return (np.maximum(dn, 0.0), up)


def entropy(a, n):
    """[lo, hi] for H(V) = sum -v log2 v over the last axis (n entries)."""
    return rsum(negxlog2x(a), 1, n)


def glog(a, floor=1e-300):
    """[lo, hi] for -log2(v) - 1/ln2, decreasing in v."""
    l = log2(a, floor)
    return (_dn(-l[1] - INV_LN2_HI), _up(-l[0] - INV_LN2_LO))


def validate_log2(rng, ntest=20000):
    """spot-check the LOG2_ULP widening against mpmath at 200 bits."""
    from mpmath import mp, mpf, log as mlog
    mp.prec = 200
    xs = np.concatenate([
        rng.uniform(1e-12, 1.0, ntest // 4),
        rng.uniform(0.999, 1.001, ntest // 4),
        2.0 ** rng.uniform(-300, 0, ntest // 4),
        rng.uniform(0.0, 1e-8, ntest - 3 * (ntest // 4))])
    lo, hi = log2((xs, xs))
    bad = 0; worst = 0.0
    ln2 = mlog(2)
    for i in range(0, len(xs), max(1, len(xs) // 4000)):
        x = xs[i]
        if x <= 0: continue
        t = mlog(mpf(float(x))) / ln2
        if not (mpf(float(lo[i])) <= t <= mpf(float(hi[i]))): bad += 1
        worst = max(worst, float(abs(t - mpf(float(np.log2(x))))) / max(abs(float(t)), 1e-300))
    return bad, worst
