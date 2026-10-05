"""Frame AUROC and paired whole-video bootstrap; no frame-IID resampling."""
import numpy as np


def normalized(raw, video):
    out = np.zeros_like(raw,dtype=np.float64)
    for v in np.unique(video):
        use = video == v
        lo, hi = raw[use].min(), raw[use].max()
        if hi > lo: out[use] = (raw[use]-lo)/(hi-lo)
    return out


class AUC:
    def __init__(self, y, score, video):
        self.order = np.argsort(score,kind="stable")
        s = np.asarray(score)[self.order]
        self.starts = np.r_[0,1+np.flatnonzero(s[1:] != s[:-1])]
        self.y = np.asarray(y,dtype=np.float64)[self.order]
        self.video = np.asarray(video,dtype=int)[self.order]

    def __call__(self, multiplicity):
        w = multiplicity[self.video]
        pos = np.add.reduceat(w*self.y,self.starts)
        neg = np.add.reduceat(w*(1-self.y),self.starts)
        denominator = pos.sum()*neg.sum()
        if denominator == 0: return float("nan")
        return float(np.sum(pos*(np.cumsum(neg)-0.5*neg))/denominator)


def target_metrics(y, raw, video, bootstrap_rng=None, repeats=2000):
    n = int(video.max())+1
    evaluators = {"R":AUC(y,raw,video),"N":AUC(y,normalized(raw,video),video)}
    per_video = np.full(n,np.nan)
    for v in range(n):
        use = video == v
        per_video[v] = AUC(y[use],raw[use],np.zeros(use.sum(),dtype=int))(np.ones(1))
    eligible = np.isfinite(per_video)
    point = {k:f(np.ones(n)) for k,f in evaluators.items()}
    point["W"] = float(per_video[eligible].mean())
    if bootstrap_rng is None: return point,per_video,None
    samples = {k:np.full(repeats,np.nan) for k in ("N","R","W")}
    for b in range(repeats):
        counts = np.bincount(bootstrap_rng.integers(0,n,n),minlength=n)
        for k,f in evaluators.items(): samples[k][b] = f(counts)
        if counts[eligible].sum():
            samples["W"][b] = np.sum(counts[eligible]*per_video[eligible])/counts[eligible].sum()
    return point,per_video,samples
