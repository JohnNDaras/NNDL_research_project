import numpy as np
from scipy.stats import norm, beta


########################  Wilson Method   ########################

def threshold_recall_wilson(
        probs_pos,
        target_recall=0.90,
        alpha=0.05,
        verbose=True):
    """
    probs_pos : 1-D array of calibrated scores for verified positives.
    Returns the smallest threshold whose (1-α) Wilson lower bound on
    recall ≥ target_recall.
    """
    ppos = np.sort(np.asarray(probs_pos, dtype=np.float64))[::-1]   # descending
    N    = ppos.size
    if N == 0:
        raise ValueError("No positive scores supplied.")

    z  = norm.ppf(1 - alpha)                   # e.g. 1.645 for 90 %, 1.96 for 95 %
    ks = np.arange(1, N + 1)                  # cumulative positives 1…N

    # Wilson lower bound for each prefix k
    phat = ks / N
    denom = 1 + z**2 / N
    centre = phat + z**2 / (2*N)
    halfwidth = z * np.sqrt((phat*(1 - phat) + z**2/(4*N)) / N)
    lb = (centre - halfwidth) / denom         # vectorised

    idx = np.argmax(lb >= target_recall)
    if lb[idx] < target_recall:               # cannot attain
        threshold = ppos[-1]
        if verbose:
            print(f"[Wilson] cannot guarantee {target_recall:.2f}; "
                  f"best ≈ {lb[-1]:.3f}")
    else:
        threshold = ppos[idx]

    if verbose:
        achieved = (ppos >= threshold).mean()
        print(f"[Wilson] threshold={threshold:.4f}  "
              f"(LB={lb[idx]:.3f}, empirical recall={achieved:.3f})")
    return float(threshold)





########################   Clopper-Pearson Method   ########################

def threshold_recall_confidence(
        probs_pos,
        target_recall=0.90,
        alpha=0.05,
        verbose=True):
    """
    `probs_pos` : 1-D array of *positive* scores (after any calibration)
    Returns the smallest threshold whose 100*(1-α)% Clopper–Pearson lower
    confidence bound for recall is ≥ `target_recall`.
    """
    ppos = np.sort(np.asarray(probs_pos, dtype=np.float64))[::-1]  # desc
    N    = ppos.size
    if N == 0:
        raise ValueError("No positive scores supplied.")

    # cumulative positives: k = 1 … N
    ks   = np.arange(1, N + 1)              # 1-based count
    # Clopper–Pearson lower bound for each k
    lb   = beta.ppf(alpha, ks, N - ks + 1)

    # first index where lower bound ≥ target recall
    idx  = np.argmax(lb >= target_recall)
    if lb[idx] < target_recall:
        # even the full set doesn’t reach desired bound → fallback to min score
        threshold = ppos[-1]
        if verbose:
            print(f"[CP-recall] Cannot guarantee {target_recall:.2f}, "
                  f"best achievable ≈ {lb[-1]:.3f}")
    else:
        threshold = ppos[idx]

    if verbose:
        print(f"[CP-recall] threshold={threshold:.4f} "
              f"(LB@idx {lb[idx]:.3f}, N={N})")
    return float(threshold)





########################   QuantCI Method   ########################

def threshold_quant_ci(scores_all, labels, target_recall=0.90,
                       alpha=0.05, verbose=True):
    order = np.argsort(-scores_all)
    scores_sorted = scores_all[order]

    # Convert labels to a NumPy array if it's a Pandas Series
    # or a DataFrame column
    if not isinstance(labels, np.ndarray):
      # Convert labels to a NumPy array if it's not already
      labels = np.array(labels)

    y_sorted      = labels[order]

    Npos = y_sorted.sum()
    if Npos == 0:
        raise ValueError("No positives")

    z = norm.ppf(1-alpha)              # 1.645 for 90 %; 1.96 for 95 %

    cum_pos = 0
    for k,(s,y) in enumerate(zip(scores_sorted, y_sorted), start=1):
        cum_pos += y
        r_hat = cum_pos / Npos
        se    = np.sqrt(r_hat*(1-r_hat) / Npos)
        lb    = r_hat - z * se
        if lb >= target_recall:
            thr = s
            if verbose:
                print(f"[QuantCI] stop@rank {k}  R̂={r_hat:.3f} LB={lb:.3f} "
                      f"→ thr={thr:.6f}")
            return float(thr)

    if verbose:
        print("[QuantCI] never hit target – return min score")
    return float(scores_sorted[-1])













