"""Per-video normalization and tie-correct pooled frame AUROC."""
import numpy as np

def normalized(raw, video):
    out = np.zeros_like(raw, dtype=np.float64)
    for v in np.unique(video):
        use = video == v
        (lo, hi) = (raw[use].min(), raw[use].max())
        if hi > lo:
            out[use] = (raw[use] - lo) / (hi - lo)
    return out

class AUC:

    def __init__(self, y, score, video):
        self.order = np.argsort(score, kind='stable')
        s = np.asarray(score)[self.order]
        self.starts = np.r_[0, 1 + np.flatnonzero(s[1:] != s[:-1])]
        self.y = np.asarray(y, dtype=np.float64)[self.order]
        self.video = np.asarray(video, dtype=int)[self.order]

    def __call__(self, multiplicity):
        w = multiplicity[self.video]
        pos = np.add.reduceat(w * self.y, self.starts)
        neg = np.add.reduceat(w * (1 - self.y), self.starts)
        denominator = pos.sum() * neg.sum()
        if denominator == 0:
            return float('nan')
        return float(np.sum(pos * (np.cumsum(neg) - 0.5 * neg)) / denominator)
