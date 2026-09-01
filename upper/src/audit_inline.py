"""Certified pass with an INLINE exhaustive float64 recheck -- no 4 GB bias file on disk.

Builds the reward in float32 (parallel), solves for the bias h, certifies, and then rebuilds
the reward chunk by chunk in float64 and recomputes theta = max over ALL (s,a) of
rbar + h(s') - h(s) with the same h.  The float64 value is the one reported.
"""
import numpy as np, sys, os, time
from multiprocessing import Process, shared_memory as shm
from src.filter_win2 import WinModel2, blocks2
from src.filter_par import parvi, parcert
from src.filter_win2 import WinModel2 as _WM
from multiprocessing import Value, Array
from src.filter_window import _apply_T


def _f64_worker(hname, n, d, L, k, m, m1f, w, ctr, nP, res, wid):
    """exhaustive float64 recheck, parallel: each worker takes prefixes off a shared counter
    and reports the max of  rbar + h(s') - h(s)  over the states it owned."""
    hsm = shm.SharedMemory(name=hname)
    hd = np.ndarray((n,), np.float64, buffer=hsm.buf)
    md = _WM(d, L, k, m)
    best = -np.inf
    while True:
        with ctr.get_lock():
            P = ctr.value
            if P >= nP:
                break
            ctr.value = P + 1
        C, W = md.start(np.float64)
        for j in range(m - 1, m1f - 1, -1):
            C, W = md.step(C, W, (P >> (j - m1f)) & 1)
        for _ in range(m1f):
            C0, W0 = md.step(C, W, 0); C1, W1 = md.step(C, W, 1)
            Cn = np.empty((2 * C.shape[0], C.shape[1]), np.float64)
            Wn = np.empty((2 * W.shape[0], W.shape[1]), np.float64)
            Cn[0::2] = C0; Cn[1::2] = C1; Wn[0::2] = W0; Wn[1::2] = W1
            C, W = Cn, Wn
        o = P << m1f
        s = np.arange(o, o + C.shape[0], dtype=np.int64)
        e = (s << 1) & (n - 1)
        for c in (0, 1):
            v = md.reward(C, W, c, w) + hd[e | c] - hd[s]
            best = max(best, float(v.max()))
    res[wid] = best
    hsm.close()


def _dyn_worker(n0, n1, n, d, L, k, m, m1, w, ctr, nP):
    """pull the next window prefix off a shared counter: dynamic load balancing, so one
    slow chunk cannot leave a straggler holding the whole build."""
    a0 = shm.SharedMemory(name=n0); a1 = shm.SharedMemory(name=n1)
    R0 = np.ndarray((n,), np.float32, buffer=a0.buf)
    R1 = np.ndarray((n,), np.float32, buffer=a1.buf)
    md = _WM(d, L, k, m)
    while True:
        with ctr.get_lock():
            P = ctr.value
            if P >= nP:
                break
            ctr.value = P + 1
        C, W = md.start(np.float32)
        for j in range(m - 1, m1 - 1, -1):
            C, W = md.step(C, W, (P >> (j - m1)) & 1)
        for _ in range(m1):
            C0, W0 = md.step(C, W, 0); C1, W1 = md.step(C, W, 1)
            Cn = np.empty((2 * C.shape[0], C.shape[1]), np.float32)
            Wn = np.empty((2 * W.shape[0], W.shape[1]), np.float32)
            Cn[0::2] = C0; Cn[1::2] = C1; Wn[0::2] = W0; Wn[1::2] = W1
            C, W = Cn, Wn
        o = P << m1
        R0[o:o + C.shape[0]] = md.reward(C, W, 0, w)
        R1[o:o + C.shape[0]] = md.reward(C, W, 1, w)
    a0.close(); a1.close()

if __name__ == '__main__':
    d = float(sys.argv[1]); L = int(sys.argv[2]); k = int(sys.argv[3]); m = int(sys.argv[4])
    z = np.load(f'results/q_d{d}_L{L}_k{k}.npz'); w = z['w']
    md = WinModel2(d, L, k, m)
    nproc = int(os.environ.get('BDC_NP', 8)); m1 = int(os.environ.get('BDC_M1', 15))
    n = 1 << m
    print(f'INLINE d={d} L={L} k={k} M={md.M} m={m} states=2^{m} pi_m={md.pi:.4e}', flush=True)
    a0 = shm.SharedMemory(create=True, size=n * 4); a1 = shm.SharedMemory(create=True, size=n * 4)
    nP = 1 << (m - m1)
    ctr = Value('i', 0)
    t0 = time.time()
    ps = [Process(target=_dyn_worker,
                  args=(a0.name, a1.name, n, d, L, k, m, m1, w, ctr, nP))
          for i in range(nproc)]
    for p in ps: p.start()
    for p in ps: p.join()
    bad = [(i, q.exitcode) for i, q in enumerate(ps) if q.exitcode != 0]
    if bad: raise RuntimeError(f'reward workers failed: {bad}')
    R0 = np.ndarray((n,), np.float32, buffer=a0.buf); R1 = np.ndarray((n,), np.float32, buffer=a1.buf)
    print(f'  reward built [{time.time()-t0:.0f}s]', flush=True)
    h = parvi(R0, R1, m, iters=int(os.environ.get('BDC_VI', 350)), nthr=nproc, verbose=True)
    th32 = parcert(R0, R1, h, m, nthr=nproc) / (1 - d)
    print(f'  float32 theta/(1-d) = {th32:.8f}   now rebuilding in float64...', flush=True)
    del R0, R1
    for a in (a0, a1): a.close(); a.unlink()
    hd = h.astype(np.float64); del h
    hs = shm.SharedMemory(create=True, size=n * 8)
    np.ndarray((n,), np.float64, buffer=hs.buf)[:] = hd
    del hd
    m1f = min(int(os.environ.get('BDC_M1F', 13)), m)
    nPf = 1 << (m - m1f)
    ctr2 = Value('i', 0)
    res = Array('d', nproc)
    t1 = time.time()
    ps2 = [Process(target=_f64_worker,
                   args=(hs.name, n, d, L, k, m, m1f, w.astype(np.float64), ctr2, nPf, res, i))
           for i in range(nproc)]
    for p in ps2: p.start()
    for p in ps2: p.join()
    bad2 = [(i, q.exitcode) for i, q in enumerate(ps2) if q.exitcode != 0]
    if bad2: raise RuntimeError(f'float64 workers failed: {bad2}')
    best = max(res[:]); tot = n
    hs.close(); hs.unlink()
    th = best / (1 - d) + 1e-4
    print(f'  float64 EXHAUSTIVE over {tot} = 2^{m} states: theta/(1-d) = {th:.8f}'
          f'   (float32 differed by {th32 + 1e-4 - th:+.2e})   [{time.time()-t1:.0f}s]', flush=True)
    print(f'  ==> C(d) <= {th:.6f} (1-d) for all d >= {d};  at d={d}: C <= {th*(1-d):.6f}')
    print(f'  atlas narrowing (LB 0.123564): {(0.3578-th+0.001464)/0.2357*100:.2f}%')
    np.savez(f'results/inline_d{d}_L{L}_k{k}_m{m}.npz', w=w, theta=th * (1 - d), d=d, L=L,
             k=k, m=m, pi=md.pi, th32=th32)
