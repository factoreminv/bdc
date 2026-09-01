"""Exact-enough Kraft check on the stored test measure.

The bound  C <= sup_x avg_i [ p <mu|_k, w(c,.)> - JS_p ]  is only valid when q is a genuine
sub-probability, i.e. for every context  2^{-w[0][ctx]} + 2^{-w[1][ctx]} <= 1.  q is
PARAMETERISED as w = [-log2(1-t), -log2(t)], so the identity holds exactly in the reals -- but
the stored w are ROUNDED float64, so the identity can fail by an ulp.  A violation makes the
certificate unsound; the repair is to RAISE w (which lowers 2^-w and can only weaken, never
invalidate, the bound).

mpmath at 300 bits decides a question whose scale is 1e-16, so this is decisive.
"""
import sys, numpy as np
from mpmath import mp, mpf, power

def check(path, prec=300, repair=False):
    mp.prec = prec
    z = np.load(path); w = z['w'].astype(np.float64)
    nctx = w.shape[1]
    worst = mpf(0); nbad = 0; bad = []
    wfix = w.copy()
    for c in range(nctx):
        s = power(2, -mpf(float(w[0, c]))) + power(2, -mpf(float(w[1, c])))
        ex = s - 1
        if ex > worst: worst = ex
        if s > 1:
            nbad += 1; bad.append((c, float(ex)))
            if repair:                       # raise BOTH entries by one ulp: 2^-w drops
                wfix[0, c] = np.nextafter(wfix[0, c], np.inf)
                wfix[1, c] = np.nextafter(wfix[1, c], np.inf)
    return nctx, nbad, float(worst), bad, wfix, w

if __name__ == '__main__':
    for path in sys.argv[1:]:
        nctx, nbad, worst, bad, wfix, w = check(path, repair=True)
        tag = 'KRAFT OK' if nbad == 0 else f'KRAFT VIOLATED in {nbad}/{nctx} contexts'
        print(f'{path}\n  {tag};  max(sum 2^-w - 1) = {worst:+.3e}  over {nctx} contexts')
        if nbad:
            print('   worst contexts:', bad[:5])
            nctx2, nbad2, worst2, _, _, _ = check(path, repair=False)
            np.savez(path.replace('.npz', '_kraft.npz'), w=wfix,
                     **{k: z for k, z in np.load(path).items() if k != 'w'})
            print(f'   repaired copy written to {path.replace(".npz","_kraft.npz")}')
