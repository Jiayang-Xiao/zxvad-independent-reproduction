"""Recompute published legacy AUROC and validate score/frame/label alignment."""
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from metrics import AUC, normalized

def main():
    root=ROOT/'evidence/legacy_baseline'
    expected=json.loads((root/'results.json').read_text())
    values=[]
    for row in expected['rows']:
        with np.load(root/(row['target']+'.npz'),allow_pickle=False) as z:
            raw,y,video,frame=z['raw'],z['label'],z['video'],z['frame']
            assert raw.ndim==1 and raw.shape==y.shape==video.shape==frame.shape
            assert np.isfinite(raw).all() and np.isin(y,[0,1]).all()
            assert len(raw)==row['frames'] and len(z['video_names'])==row['videos']
            assert np.array_equal(np.unique(video),np.arange(row['videos']))
            for v in np.unique(video):
                indices=frame[video==v]
                assert np.array_equal(indices,np.arange(4,4+len(indices))), 'Frame/label alignment changed'
            value=AUC(y,normalized(raw,video),video)(np.ones(row['videos']))
            assert abs(value-row['AUROC'])<1e-14,(row['target'],value,row['AUROC'])
            values.append(value)
            print(f"{row['target']}: AUROC {100*value:.4f}% ({row['frames']} frames)")
    assert abs(float(np.mean(values))-expected['macro_AUROC'])<1e-14
    # Ties and constant-video behavior must have the usual half-credit interpretation.
    assert AUC(np.array([0,1]),np.array([.5,.5]),np.zeros(2,dtype=int))(np.ones(1))==.5
    assert np.array_equal(normalized(np.array([4.,4.]),np.zeros(2,dtype=int)),np.zeros(2))
    print(f"Legacy macro AUROC {100*np.mean(values):.4f}%: PASS")

if __name__=='__main__': main()
