"""Export a completed fresh legacy_v1 fit without checkpoints/images/private bootstrap."""
import argparse
import json
import shutil
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from common import code_identity, digest, sha
from metrics import AUC, normalized

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',required=True); args=ap.parse_args()
    out=Path(args.output).resolve(); dst=ROOT/'evidence/fresh_legacy_v1_run'
    summary=json.loads((out/'summary.json').read_text()); freeze=json.loads((out/'freeze.json').read_text())
    done=json.loads((out/'baseline/completed.json').read_text()); config=json.loads((out/'baseline/config.json').read_text())
    preflight=json.loads((out/'preflight.json').read_text())
    if (out/'last_exit.txt').read_text().strip()!='0' or summary['status']!='COMPLETED': raise RuntimeError('Run has not completed successfully')
    if done['step']!=5000 or done['checkpoint_sha256']!=sha(out/'baseline/last.pt') or done['config_hash']!=digest(config): raise RuntimeError('Checkpoint/config identity mismatch')
    if config['code_sha256']!=code_identity() or freeze['code']!=code_identity(): raise RuntimeError('Current code changed since run')
    if preflight['status']!='PASS' or freeze['preflight_sha256']!=sha(out/'preflight.json'): raise RuntimeError('Preflight identity mismatch')
    if summary['freeze_sha256']!=sha(out/'freeze.json'): raise RuntimeError('Summary identity mismatch')
    if len(summary['rows'])!=3: raise RuntimeError('Expected three target AUROC rows')
    for row in summary['rows']:
        path=out/'baseline'/(row['target']+'.npz'); meta=json.loads(path.with_suffix('.json').read_text())
        if meta['freeze_sha256']!=summary['freeze_sha256'] or meta['scores_sha256']!=sha(path): raise RuntimeError('Raw score identity mismatch')
        with np.load(path,allow_pickle=False) as z:
            value=AUC(z['label'],normalized(z['raw'],z['video']),z['video'])(np.ones(int(z['video'].max())+1))
        if abs(value-row['AUROC'])>1e-14: raise RuntimeError('AUROC mismatch')
    if dst.exists():
        old=json.loads((dst/'summary.json').read_text())
        if old['freeze_sha256']!=summary['freeze_sha256']: raise RuntimeError('A different fresh run is already exported; use a new version/directory')
    dst.mkdir(parents=True,exist_ok=True)
    names=['summary.json','results.csv','freeze.json','preflight.json','environment.json','last_exit.txt',
        'baseline/config.json','baseline/completed.json','baseline/state.json','baseline/train.jsonl']
    names += [f'baseline/{target}.{ext}' for target in ('ped1','ped2','avenue') for ext in ('npz','json')]
    for name in names:
        target=dst/name; target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(out/name,target)
    (dst/'export_manifest.json').write_text(json.dumps({name:sha(dst/name) for name in names},indent=2)+'\n')
    (dst/'README.md').write_text('Fresh run of the packaged legacy_v1 baseline. This is not the corrected paper-alignment version.\n'
        'Checkpoint/image files and private bootstrap samples remain on the server.\n',encoding='utf-8')
    print('Fresh evidence exported: '+str(dst))

if __name__=='__main__': main()
