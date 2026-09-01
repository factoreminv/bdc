"""CERTIFIED pass: float32 build + value iteration, then a RIGOROUS interval-arithmetic
exhaustive maximum.  The reported theta is a proved upper bound, not a float with a slack.

    C(d) <= theta_hi = max over ALL (s,a) of  [ rbar_ub(s,a) + h(s') - h(s) ]

rbar_ub comes from src/rigor.py (directed rounding throughout, gradient carried as an
enclosure so concavity is applied to a genuine supergradient, d seeded as an interval around
the REAL rational, pi_m exact).  h is used exactly as stored -- it is arbitrary, so its float
values are exact rationals and contribute no error.  The test measure must satisfy Kraft
exactly (src/kraft.py); pass the repaired file.
"""
import numpy as np, sys, os, time
from fractions import Fraction
from multiprocessing import Process, Value, Array, shared_memory as shm
from src.filter_win2 import WinModel2
from src.filter_par import parvi
from src.rigor import IvModel
from src import ivl
from src.audit_inline import _dyn_worker


def _iv_worker(hname, hn, n, dnum, dden, L, k, m, m1, w, ctr, nP, res, wid):
    hsm = shm.SharedMemory(name=hname)
    hd = np.ndarray((hn,), np.float64, buffer=hsm.buf); hmask = hn - 1
    md = IvModel(Fraction(dnum, dden), L, k, m)
    best = -np.inf
    while True:
        with ctr.get_lock():
            P = ctr.value
            if P >= nP: break
            ctr.value = P + 1
        C, W = md.start()
        for j in range(m - 1, m1 - 1, -1):
            C, W = md.step(C, W, (P >> (j - m1)) & 1)
        for _ in range(m1):
            C0, W0 = md.step(C, W, 0); C1, W1 = md.step(C, W, 1)
            def il(a, b):
                o0 = np.empty((2 * a[0].shape[0], a[0].shape[1])); o0[0::2] = a[0]; o0[1::2] = b[0]
                o1 = np.empty((2 * a[1].shape[0], a[1].shape[1])); o1[0::2] = a[1]; o1[1::2] = b[1]
                return (o0, o1)
            C, W = il(C0, C1), il(W0, W1)
        o = P << m1
        s = np.arange(o, o + C[0].shape[0], dtype=np.int64)
        e = (s << 1) & (n - 1)
        for c in (0, 1):
            v = np.nextafter(md.rbar_ub(C, W, c, w) +
                             np.nextafter(hd[(e | c) & hmask] - hd[s & hmask], np.inf),
                             np.inf)
            best = max(best, float(v.max()))
    res[wid] = best
    hsm.close()


if __name__ == '__main__':
    dnum, dden = int(sys.argv[1]), int(sys.argv[2])
    L, k, m = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    qpath = sys.argv[6]
    d_frac = Fraction(dnum, dden); d = float(d_frac)
    z = np.load(qpath); w = z['w'].astype(np.float64)
    # refuse to run on a measure that is not a genuine sub-probability
    from src.kraft import check
    nctx, nbad, worst, _, _, _ = check(qpath)
    if nbad: raise SystemExit(f'REFUSING: Kraft violated in {nbad}/{nctx} contexts ({worst:+.2e})')
    print(f'Kraft OK on {qpath}: max(sum 2^-w - 1) = {worst:+.2e} over {nctx} contexts', flush=True)

    md = WinModel2(d, L, k, m); iv = IvModel(d_frac, L, k, m)
    nproc = int(os.environ.get('BDC_NP', 8)); m1 = int(os.environ.get('BDC_M1', 15))
    n = 1 << m
    print(f'RIGOR d={d_frac}={d} L={L} k={k} M={md.M} m={m} states=2^{m} '
          f'pi_m(exact,up)={iv.pi_up:.6e}', flush=True)
    bias_path = os.environ.get('BDC_BIAS')
    if bias_path:
        hb = np.load(bias_path).astype(np.float64)
        if hb.ndim != 1 or hb.size > n or (hb.size & (hb.size - 1)):
            raise ValueError('BDC_BIAS must be a power-of-two one-dimensional bias no longer than 2^m')
        h = hb
        print(f'  lifted bias {bias_path}: 2^{int(np.log2(hb.size))} -> 2^{m}', flush=True)
    else:
        a0 = shm.SharedMemory(create=True, size=n * 4); a1 = shm.SharedMemory(create=True, size=n * 4)
        ctr = Value('i', 0); nP = 1 << (m - m1)
        t0 = time.time()
        ps = [Process(target=_dyn_worker,
                      args=(a0.name, a1.name, n, d, L, k, m, m1, w.astype(np.float32), ctr, nP))
              for _ in range(nproc)]
        for p in ps: p.start()
        for p in ps: p.join()
        if [q for q in ps if q.exitcode != 0]: raise RuntimeError('reward workers failed')
        R0 = np.ndarray((n,), np.float32, buffer=a0.buf); R1 = np.ndarray((n,), np.float32, buffer=a1.buf)
        print(f'  reward built [{time.time()-t0:.0f}s]', flush=True)
        h = parvi(R0, R1, m, iters=int(os.environ.get('BDC_VI', 350)), nthr=nproc, verbose=True)
        del R0, R1
        for a in (a0, a1): a.close(); a.unlink()
        bias_out = f'results/bias_d{d}_L{L}_k{k}_m{m}.npy'
        np.save(bias_out, h)
        print(f'  bias saved to {bias_out}', flush=True)
        if os.environ.get('BDC_STOP_AFTER_BIAS') == '1':
            raise SystemExit(0)

    hn = h.size
    hs = shm.SharedMemory(create=True, size=hn * 8)
    np.ndarray((hn,), np.float64, buffer=hs.buf)[:] = h.astype(np.float64)
    del h
    m1f = min(int(os.environ.get('BDC_M1F', 13)), m)
    ctr2 = Value('i', 0); res = Array('d', nproc); nPf = 1 << (m - m1f)
    t1 = time.time()
    ps2 = [Process(target=_iv_worker,
                   args=(hs.name, hn, n, dnum, dden, L, k, m, m1f, w, ctr2, nPf, res, i))
           for i in range(nproc)]
    for p in ps2: p.start()
    for p in ps2: p.join()
    if [q for q in ps2 if q.exitcode != 0]: raise RuntimeError('interval workers failed')
    best = max(res[:])
    hs.close(); hs.unlink()
    th = np.nextafter(best / (1 - d), np.inf)
    print(f'  INTERVAL EXHAUSTIVE over all 2^{m} states: theta_hi/(1-d) = {th:.9f}  '
          f'[{time.time()-t1:.0f}s]', flush=True)
    print(f'  ==> C(d) <= {th:.6f} (1-d) for all d >= {d_frac};  at d={d}: C <= {th*(1-d):.6f}')
    print(f'  atlas narrowing (LB 0.123564): {(0.3578-th+0.001464)/0.2357*100:.2f}%')
    np.savez(f'results/rigor_d{d}_L{L}_k{k}_m{m}.npz', theta_hi=th, d=d, L=L, k=k, m=m,
             pi_up=iv.pi_up, w=w)
