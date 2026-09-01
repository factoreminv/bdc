"""Parallel reward build (processes over window prefixes, shared memory) and threaded
average-reward VI, for the 2^30 - 2^32 filter-window certificates."""
import numpy as np, sys, os, time
from multiprocessing import Process
from multiprocessing import shared_memory as shm
from concurrent.futures import ThreadPoolExecutor
from src.filter_window import WinModel, blocks


def _worker(name0, name1, n, d, L, k, m, m1, w, plo, phi):
    a0 = shm.SharedMemory(name=name0); a1 = shm.SharedMemory(name=name1)
    R0 = np.ndarray((n,), np.float32, buffer=a0.buf)
    R1 = np.ndarray((n,), np.float32, buffer=a1.buf)
    md = WinModel(d, L, k, m)
    dd, MM = md.d, md.M
    from src.filter_window import _apply_T
    for P in range(plo, phi):
        v = np.full((1, 1 << MM), 2.0 ** -MM, dtype=np.float32)
        for j in range(m - 1, m1 - 1, -1):
            v = _apply_T(v, (P >> (j - m1)) & 1, dd, MM)
        A = v
        for _ in range(m1):
            B = np.empty((2 * A.shape[0], 1 << MM), np.float32)
            B[0::2] = _apply_T(A, 0, dd, MM)
            B[1::2] = _apply_T(A, 1, dd, MM)
            A = B
        o = P << m1
        R0[o:o + A.shape[0]] = md.reward(A, 0, w)
        R1[o:o + A.shape[0]] = md.reward(A, 1, w)
    a0.close(); a1.close()


def parbuild(d, L, k, m, w, m1=21, nproc=8):
    n = 1 << m
    a0 = shm.SharedMemory(create=True, size=n * 4)
    a1 = shm.SharedMemory(create=True, size=n * 4)
    nP = 1 << (m - m1)
    bounds = [(i * nP) // nproc for i in range(nproc + 1)]
    ps = [Process(target=_worker, args=(a0.name, a1.name, n, d, L, k, m, m1, w,
                                        bounds[i], bounds[i + 1])) for i in range(nproc)]
    for p in ps: p.start()
    for p in ps: p.join()
    bad = [(i, q.exitcode) for i, q in enumerate(ps) if q.exitcode != 0]
    if bad:
        raise RuntimeError(f'reward workers failed: {bad} -- shared memory would be uninitialised')
    R0 = np.ndarray((n,), np.float32, buffer=a0.buf)
    R1 = np.ndarray((n,), np.float32, buffer=a1.buf)
    return R0, R1, (a0, a1)


def parvi(R0, R1, m, iters=250, nthr=8, verbose=True):
    n = 1 << m
    half = n >> 1
    h = np.zeros(n, np.float32)
    g = np.empty(n, np.float32)
    cuts = [(i * n) // nthr for i in range(nthr + 1)]
    cuts = sorted(set(cuts + [half]))
    seg = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]

    def step(lo, hi, src, dst):
        b = lo - (half if lo >= half else 0)
        e = b + (hi - lo)
        x = src[2 * b:2 * e:2]; y = src[2 * b + 1:2 * e:2]
        np.add(R0[lo:hi], x, out=dst[lo:hi])
        t = np.add(R1[lo:hi], y)
        np.maximum(dst[lo:hi], t, out=dst[lo:hi])

    t0 = time.time()
    with ThreadPoolExecutor(nthr) as ex:
        for it in range(iters):
            list(ex.map(lambda s: step(s[0], s[1], h, g), seg))
            c = g[0]
            list(ex.map(lambda s: np.subtract(g[s[0]:s[1]], c, out=g[s[0]:s[1]]), seg))
            h, g = g, h
            if verbose and it % 25 == 24:
                print(f'      VI {it+1}/{iters}  [{time.time()-t0:.0f}s]', flush=True)
    return h


def parcert(R0, R1, h, m, nthr=8):
    n = 1 << m; half = n >> 1
    cuts = sorted(set([(i * n) // nthr for i in range(nthr + 1)] + [half]))
    best = [-1e30] * (len(cuts) - 1)

    def f(i):
        lo, hi = cuts[i], cuts[i + 1]
        b = lo - (half if lo >= half else 0); e = b + (hi - lo)
        v = R0[lo:hi] + h[2 * b:2 * e:2]
        np.maximum(v, R1[lo:hi] + h[2 * b + 1:2 * e:2], out=v)
        v -= h[lo:hi]
        best[i] = float(v.max())
    with ThreadPoolExecutor(nthr) as ex:
        list(ex.map(f, range(len(cuts) - 1)))
    return max(best)


if __name__ == '__main__':
    d = float(sys.argv[1]); L = int(sys.argv[2]); k = int(sys.argv[3]); m = int(sys.argv[4])
    z = np.load(f'results/q_d{d}_L{L}_k{k}.npz'); w = z['w']
    md = WinModel(d, L, k, m)
    print(f'PAR d={d} L={L} k={k} M={md.M} m={m} states=2^{m} pi_m={md.pi:.4e} '
          f'(q lib value {float(z["cert"]):.5f})', flush=True)
    nproc = int(os.environ.get('BDC_NP', 8)); m1 = int(os.environ.get('BDC_M1', 21))
    t0 = time.time()
    R0, R1, keep = parbuild(d, L, k, m, w, m1=m1, nproc=nproc)
    print(f'  reward built [{time.time()-t0:.0f}s]', flush=True)
    h = parvi(R0, R1, m, iters=int(os.environ.get('BDC_VI', 250)), nthr=nproc)
    th = parcert(R0, R1, h, m, nthr=nproc) / (1 - d)
    SLACK = 1e-4      # float32 reward table: measured |f32 - f64| <= 7e-7 bits/symbol
    th += SLACK
    print(f'  CERTIFIED /(1-d) = {th:.6f}  (incl. {SLACK} float32 slack)   '
          f'at d={d}: C <= {th*(1-d):.6f}', flush=True)
    print(f'  atlas narrowing of [0.1221,0.3578] with LB 0.123564: '
          f'{(0.3578-th+0.001464)/0.2357*100:.2f}%', flush=True)
    np.savez(f'results/par_d{d}_L{L}_k{k}_m{m}.npz', w=w, theta=th * (1 - d), d=d, L=L, k=k,
             m=m, pi=md.pi)
    for a in keep:
        a.close(); a.unlink()
