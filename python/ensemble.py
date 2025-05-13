########################    inverse‑variance ensemble on ONE stratified subsample   ########################

import numpy as np
from scipy.stats import beta, norm
from math import sqrt

# ---------- 4 individual threshold routines (short forms) ----------------
def thr_cp(pos_scores, tgt, a=0.05):
    N = len(pos_scores); ks = np.arange(1, N+1)
    lb = beta.ppf(a, ks, N-ks+1); idx = np.argmax(lb >= tgt)
    return pos_scores[::-1][idx] if lb[idx] >= tgt else pos_scores.min()

def thr_jf(pos_scores, tgt, a=0.05):
    N = len(pos_scores); ks = np.arange(1, N+1)
    lb = beta.ppf(a, ks+.5, N-ks+.5); idx = np.argmax(lb >= tgt)
    return pos_scores[::-1][idx] if lb[idx] >= tgt else pos_scores.min()

def thr_ws(pos_scores, tgt, a=0.05):
    N = len(pos_scores); ks = np.arange(1, N+1); z = norm.ppf(1-a)
    ph = ks/N
    lb = ((ph + z*z/(2*N)) - z*np.sqrt((ph*(1-ph)+z*z/(4*N))/N)) / (1+z*z/N)
    idx = np.argmax(lb >= tgt)
    return pos_scores[::-1][idx] if lb[idx] >= tgt else pos_scores.min()

def thr_exact(pos_scores, tgt):
    N = len(pos_scores)
    idx = (1-tgt)*(N-1); lo, hi = int(np.floor(idx)), int(np.ceil(idx))
    w = idx-lo; return (1-w)*pos_scores[lo] + w*pos_scores[hi]


# ---------------- ensemble with bootstrap --------------------------------
def ensemble_threshold(pos_scores,
                       target_recall=0.90,
                       alpha        =0.05,
                       n_boot       =200,
                       random_state =42,
                       verbose=True):
    """
    pos_scores : 1‑D ndarray of calibrated scores for positives (sorted asc/desc ok)
    Returns an inverse‑variance weighted average of CP, Jeffreys, Wilson, Exact.
    """
    rng  = np.random.default_rng(random_state)
    pos_scores = np.sort(np.asarray(pos_scores, float))  # ascending


    methods = {
        'CP'   : lambda s: thr_cp(s, target_recall, alpha),
        'JF'   : lambda s: thr_jf(s, target_recall, alpha),
        'WL'   : lambda s: thr_ws(s, target_recall, alpha),
        'EX'   : lambda s: thr_exact(s, target_recall)
    }

    # bootstrapped thresholds for each method
    th_dict, var_dict = {}, {}
    for name, func in methods.items():
        ths = []
        for _ in range(n_boot):
            samp = rng.choice(pos_scores, size=pos_scores.size, replace=True)
            ths.append(func(np.sort(samp)))
        ths = np.array(ths)
        th_dict[name] = ths.mean()
        var_dict[name] = ths.var(ddof=1) + 1e-9        # avoid /0
        if verbose:
            print(f"{name:>2}  μ={ths.mean():.6f}  σ={ths.std(ddof=1):.6f}")

    # inverse‑variance weighting  w_i ∝ 1/σ_i²
    inv_var = np.array([1/var_dict[n] for n in methods])
    weights = inv_var / inv_var.sum()
    ensemble_thr = float(np.dot(weights, [th_dict[n] for n in methods]))

    if verbose:
        w_tab = ", ".join(f"{n}:{weights[i]:.2f}" for i,n in enumerate(methods))
        print(f"weights → {w_tab}")
        print(f"[Ensemble] threshold={ensemble_thr:.6f}")

    return ensemble_thr

# ---------------- multi ensemble with bootstrap --------------------------------
def ensemble_threshold_multi(pos_scores,
                             target_recall=0.90,
                             alpha        =0.05,
                             n_boot       =200,
                             K            =9,
                             subsample_fr =0.80,    # 80 % without replacement
                             random_state =42,
                             fuse_method  ="min",
                             verbose=True):
    """
    • Draw K independent subsamples of the positive scores
      (size = subsample_fr × N, no replacement → low overlap).
    • Run the inverse‑variance ensemble on each subsample.
    • Fuse the K thresholds with either 'median' (default) or
      inverse‑variance weighting across subsamples.

    Returns
    -------
    final_thr : float
        Recommended probability threshold.
    all_thrs  : list[float]
        Thresholds from each subsample (for diagnostics).
    """
    rng  = np.random.default_rng(random_state)
    pos_scores = np.asarray(pos_scores, float)

    N  = pos_scores.size
    m  = max(10, int(subsample_fr * N))         # at least 10 scores
    all_thrs, all_vars = [], []

    for k in range(K):
        # random subsample without replacement
        idx  = rng.choice(N, size=m, replace=False)
        subs = pos_scores[idx]
        thr_k = ensemble_threshold(
                    subs,
                    target_recall=target_recall,
                    alpha       =alpha,
                    n_boot      =n_boot,
                    random_state=random_state + k,
                    verbose     =False)
        all_thrs.append(thr_k)
        # quick variance estimate across n_boot*len(methods) values
        # assume ensemble std ≈ max single‑method std / √len(methods)
        all_vars.append( (0.0005)**2 )          # conservative fallback
        if verbose:
            print(f"subsample {k+1}/{K}  → threshold={thr_k:.6f}")

    all_thrs = np.array(all_thrs)
    all_vars = np.array(all_vars)  # here constant, but keep for future

    # ----- fuse -------------------------------------------------------------
    if fuse_method == "median":
        final_thr = float(np.median(all_thrs))
    elif fuse_method == "mean":
        final_thr = float(np.mean(all_thrs))
    elif fuse_method == "min":
        final_thr = float(np.min(all_thrs))
    else:  # inverse‑variance across subsamples
        w = 1 / all_vars
        w /= w.sum()
        final_thr = float(np.dot(w, all_thrs))

    if verbose:
        if fuse_method == "median":
            print(f"[Multi] median of {K} thresholds = {final_thr:.6f}")
        else:
            print(f"[Multi] inverse‑var fused threshold = {final_thr:.6f}")

    return final_thr, all_thrs.tolist()



