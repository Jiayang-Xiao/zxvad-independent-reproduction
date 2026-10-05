"""AUROC-only report with an optional private video-bootstrap archive."""
import argparse
import csv
import io
import json
from pathlib import Path
import numpy as np
from common import TARGETS, EXPECTED, save_json, sha
from metrics import AUC, normalized

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',required=True)
    ap.add_argument('--private-bootstrap-repeats',type=int,default=0)
    args=ap.parse_args(); out=Path(args.output)
    freeze=json.loads((out/'freeze.json').read_text())
    rows=[]; private={}; rng=np.random.default_rng(20261005)
    for target in TARGETS:
        path=out/'baseline'/(target+'.npz'); meta=json.loads(path.with_suffix('.json').read_text())
        if meta['freeze_sha256']!=sha(out/'freeze.json') or meta['scores_sha256']!=sha(path): raise RuntimeError('Score identity mismatch')
        with np.load(path,allow_pickle=False) as z:
            raw,y,video=z['raw'],z['label'],z['video']
            if len(raw)!=EXPECTED[target][1] or len(np.unique(video))!=EXPECTED[target][0]: raise RuntimeError('Frame count mismatch')
            evaluator=AUC(y,normalized(raw,video),video)
            value=evaluator(np.ones(EXPECTED[target][0]))
            if not np.isfinite(value): raise RuntimeError('Undefined AUROC')
            rows.append({'version':'legacy_v1','seed':17,'target':target,'frames':len(raw),'AUROC':value})
            if args.private_bootstrap_repeats:
                n=EXPECTED[target][0]
                private[target]=np.asarray([evaluator(np.bincount(rng.integers(0,n,n),minlength=n))
                    for _ in range(args.private_bootstrap_repeats)])
    if private:
        dst=out/'private'; dst.mkdir(exist_ok=True)
        np.savez_compressed(dst/'bootstrap_samples.npz',**private)
    stream=io.StringIO(); writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    (out/'results.csv').write_text(stream.getvalue(),encoding='utf-8')
    summary={'status':'COMPLETED','version':'legacy_v1','official_reproduction':False,
        'metric':'per-video-minmax pooled frame AUROC','macro_AUROC':float(np.mean([r['AUROC'] for r in rows])),
        'rows':rows,'freeze_sha256':sha(out/'freeze.json'),'source_only_training':True,
        'targets_development_observed':True,'checkpoint_step':5000}
    save_json(out/'summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__': main()
