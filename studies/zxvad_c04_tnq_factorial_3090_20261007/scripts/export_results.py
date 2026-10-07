"""Bundle only code, protocol and completed raw score evidence, excluding images/weights."""
import hashlib,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from common import read_json,sha,write_json,release_identity,digest
from report import generate_report
from model import CANDIDATES

def export():
    generate_report()
    out=ROOT/'outputs'; summary=read_json(out/'summary.json')
    if summary['status']!='COMPLETED' or len(summary['rows'])!=24:raise RuntimeError('All eight fits/twenty-four rows required')
    freeze=read_json(out/'freeze.json'); prepared=read_json(out/'prepared.json'); preflight=read_json(out/'preflight.json')
    if prepared['release']!=release_identity() or freeze['prepared_sha256']!=sha(out/'prepared.json') or freeze['preflight_sha256']!=sha(out/'preflight.json'):raise RuntimeError('Prepared/preflight evidence changed')
    if preflight['status']!='PASS' or preflight['prepared_sha256']!=sha(out/'prepared.json') or prepared['plan_sha256']!=sha(out/'source_plan.npy'):raise RuntimeError('Source preflight/plan mismatch')
    if prepared['data']['source_metadata_sha256']!=digest(read_json(out/'source_manifest.json')):raise RuntimeError('Source manifest mismatch')
    for c in CANDIDATES:
        completed=read_json(out/c/'completed.json');config=read_json(out/c/'config.json')
        if completed!=freeze['models'][c] or completed['step']!=5000 or completed['checkpoint_sha256']!=sha(out/c/'last.pt') or completed['config_hash']!=digest(config):raise RuntimeError('Completed checkpoint binding changed: '+c)
        if config['prepared_sha256']!=freeze['prepared_sha256'] or config['preflight_sha256']!=freeze['preflight_sha256'] or config['release']!=freeze['release']:raise RuntimeError('Fit identity mismatch: '+c)
    manifest=read_json(ROOT/'release_manifest.json')
    members={n:ROOT/n for n in manifest['files']}
    members['release_manifest.json']=ROOT/'release_manifest.json'
    evidence=['prepared.json','preflight.json','source_plan.npy','source_manifest.json','freeze.json','summary.json','results.csv','report.md','gpu_compatibility.json','pip_freeze.txt']
    for name in evidence:members['evidence/'+name]=out/name
    for c in CANDIDATES:
        for name in ['config.json','completed.json','state.json','train.jsonl']+[t+e for t in ('ped1','ped2','avenue') for e in ('.npz','.json')]:members[f'evidence/{c}/{name}']=out/c/name
        for path in sorted((out/c).glob('uncommitted_train_*.jsonl')):members[f'evidence/{c}/{path.name}']=path
    members['data_preparation/import_record.json']=ROOT/'data_view/import_record.json'
    members['data_preparation/data_config.json']=ROOT/'data_config.json'
    hashes={name:sha(path) for name,path in members.items()}
    export_manifest={'kind':'ZXVAD_C04_TNQ_FACTORIAL_CODE_AND_SCORES','release_identity':release_identity(),'summary_sha256':sha(out/'summary.json'),'files':hashes,'excluded':'Dataset images, raw videos, environment packages, checkpoints, credentials and private email'}
    temp=out/'review_bundle.zip.tmp'
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,strict_timestamps=False) as z:
        for name,path in sorted(members.items()):z.write(path,name)
        z.writestr('export_manifest.json',json.dumps(export_manifest,indent=2)+'\n')
    temp.replace(out/'review_bundle.zip')
    write_json(out/'export_record.json',{'bundle_sha256':sha(out/'review_bundle.zip'),'summary_sha256':sha(out/'summary.json'),'files':len(members)})
    print('Review bundle:',out/'review_bundle.zip',flush=True)

if __name__=='__main__':export()
