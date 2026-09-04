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
* log2 : no libm transcendental is used.  IEEE ``frexp`` gives exact power-of-two range
  reduction.  A 16384-entry table is generated at import time from exact rational bounds for
  the atanh series, and a one-term residual series has an explicit positive remainder.
  Every runtime arithmetic operation is widened outward below.
"""
import numpy as np
from fractions import Fraction

U = 2.0 ** -53


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


def _frac_outward(q):
    """Convert an exact Fraction to enclosing binary64 endpoints by exact comparison."""
    x = np.float64(float(q))
    lo = hi = x
    while Fraction.from_float(float(lo)) > q:
        lo = np.nextafter(lo, -np.inf)
    while Fraction.from_float(float(hi)) < q:
        hi = np.nextafter(hi, np.inf)
    return lo, hi


def _ln_rational_bounds(x, n=30):
    """Exact rational bounds for ln(x), 1 <= x <= 2, from the atanh series."""
    z = (x - 1) / (x + 1)
    z2 = z * z
    term = z
    s = Fraction(0)
    for k in range(n + 1):
        s += 2 * term / (2 * k + 1)
        term *= z2
    # Remaining denominators are at least 2n+3; sum the geometric majorant exactly.
    rem = 2 * term / ((2 * n + 3) * (1 - z2)) if z else Fraction(0)
    return s, s + rem


# Exact-rational construction of bounds for 1/ln(2) and a fine dyadic log table.
_LN2_LO_Q, _LN2_HI_Q = _ln_rational_bounds(Fraction(2))
_INV_LN2_Q = (Fraction(1, 1) / _LN2_HI_Q, Fraction(1, 1) / _LN2_LO_Q)
INV_LN2_LO = _frac_outward(_INV_LN2_Q[0])[0]
INV_LN2_HI = _frac_outward(_INV_LN2_Q[1])[1]
_TABLE_BITS = 14
_TABLE_N = 1 << _TABLE_BITS
_LOGC_LO = np.empty(_TABLE_N, dtype=np.float64)
_LOGC_HI = np.empty(_TABLE_N, dtype=np.float64)
for _j in range(_TABLE_N):
    _c = Fraction(_TABLE_N + _j, _TABLE_N)
    _lc_lo, _lc_hi = _ln_rational_bounds(_c)
    _q_lo = _lc_lo / _LN2_HI_Q
    _q_hi = _lc_hi / _LN2_LO_Q
    _LOGC_LO[_j] = _frac_outward(_q_lo)[0]
    _LOGC_HI[_j] = _frac_outward(_q_hi)[1]

# For c=1+floor(N(m-1))/N and 1<=m<2, z=(m-c)/(m+c) lies in
# [0,1/(2N)].  One retained term is 2z; the omitted positive tail starts at z^3/3.
_ZMAX_Q = Fraction(1, 2 * _TABLE_N)
_RESID_REM_HI = _frac_outward(
    2 * _ZMAX_Q ** 3 / (3 * (1 - _ZMAX_Q ** 2))
)[1]
_RESID_COEFF = (_frac_outward(Fraction(2)),)


def _log2_point(x):
    """Outward log2 enclosure for positive binary64 values, without libm log."""
    x = np.asarray(x, dtype=np.float64)
    m, e = np.frexp(x)                 # exact: x = m*2^e, 1/2 <= m < 1
    m = m * 2.0
    e = e.astype(np.int64) - 1
    # Bias the bin coordinate downward.  At an exact table boundary this deliberately uses
    # the preceding center; just below a boundary it prevents a rounded product from choosing
    # c>m.  The residual bound therefore permits equality at its stated dyadic maximum.
    coord = np.nextafter((m - 1.0) * _TABLE_N, -np.inf)
    j = np.floor(coord).astype(np.int64)
    j = np.clip(j, 0, _TABLE_N - 1)
    c = 1.0 + j.astype(np.float64) / _TABLE_N

    # m and c are dyadic and the subtraction is exact by Sterbenz, but m + c is NOT
    # exact in general: m carries 52 fractional bits and the sum lands in [2,4), so one
    # bit is lost.  The quotient therefore carries TWO roundings, and a one-ulp widening
    # does not cover it -- measured, _dn(num/den) exceeds the true z for ~0.4% of inputs.
    # Widen by two units in each direction.
    num = m - c
    den = m + c
    zlo = np.maximum(_dnk(num / den, 2), 0.0)
    zhi = _upk(num / den, 2)
    z = (zlo, zhi)
    z2 = mul_nn(z, z)
    power = z
    series = (np.zeros_like(m), np.zeros_like(m))
    for k in range(1):
        series = add(series, smul_nn(_RESID_COEFF[k], power))
        power = mul_nn(power, z2)
    ln_lo = np.maximum(series[0], 0.0)
    ln_hi = _up(series[1] + _RESID_REM_HI)
    residual = mul_nn((ln_lo, ln_hi), (INV_LN2_LO, INV_LN2_HI))
    ef = e.astype(np.float64)
    # Add the integer exponent to the table value first.  This avoids forming a value near
    # one and then cancelling it when x is just below one; each addition is widened once.
    base_lo = _dn(ef + _LOGC_LO[j])
    base_hi = _up(ef + _LOGC_HI[j])
    lo = _dn(base_lo + residual[0])
    hi = _up(base_hi + residual[1])
    return lo, hi


def log2(a, floor=1e-300):
    """log2 of a non-negative interval, clamped below at `floor` (monotone increasing)."""
    same_endpoint = a[0] is a[1]
    lo = np.maximum(a[0], np.float64(floor))
    if same_endpoint:
        return _log2_point(lo)
    hi = np.maximum(a[1], np.float64(floor))
    return _log2_point(lo)[0], _log2_point(hi)[1]


def negxlog2x(a):
    """elementwise bounds on f(x) = -x log2 x over x in [lo, hi] <= 1.

    f is concave with its maximum at x = 1/e; f(0) = 0."""
    lo_, hi_ = np.maximum(a[0], 0.0), np.maximum(a[1], 0.0)
    # Enclose log2 of each endpoint once, then reuse both sides.
    l_lo = log2((lo_, lo_))
    l_hi = log2((hi_, hi_))
    fa_u = _up(-lo_ * l_lo[0])
    fb_u = _up(-hi_ * l_hi[0])
    fa_d = _dn(-lo_ * l_lo[1])
    fb_d = _dn(-hi_ * l_hi[1])
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
    """Diagnostic comparison of the proved enclosure against mpmath at 200 bits."""
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
        mid = (float(lo[i]) + float(hi[i])) / 2
        worst = max(worst, float(abs(t - mpf(mid))) / max(abs(float(t)), 1e-300))
    return bad, worst
