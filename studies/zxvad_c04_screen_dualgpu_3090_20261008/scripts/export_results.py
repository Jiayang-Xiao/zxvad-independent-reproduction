"""Export only completed, bound two-GPU evidence. Never publish model weights."""
import json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from common import read_json,sha,write_json,release_identity,digest
from report import generate_report
from model import ORDER
from schedule import QUEUES,STUDY,lane_for
def export():
    out=ROOT/'outputs';release=release_identity()
    receipt=read_json(out/'run_completion.json')
    if receipt['status']!='ALL_GPU_WORK_COMPLETED' or receipt['queues']!=QUEUES or receipt['wall_time_limit'] is not None or receipt['worker_exits']!={p:{k:0 for k in QUEUES} for p in ('preflight','fit','evaluate')}:raise RuntimeError('Incomplete or failed GPU work')
    prepared=read_json(out/'prepared.json')
    if prepared['release']!=release or prepared['plan_sha256']!=sha(out/'source_plan.npy') or prepared['view_record_sha256']!=sha(ROOT/'data_view/import_record.json') or prepared['data']['source_metadata_sha256']!=digest(read_json(out/'source_manifest.json')):raise RuntimeError('Prepared dataset/source plan changed')
    bindings={k:sha(out/'workers'/('gpu'+k)/'preflight.json') for k in QUEUES}
    parity=read_json(out/'cross_card_parity.json')
    if parity['status']!='PASS' or parity['worker_preflights']!=bindings:raise RuntimeError('Cross-card source parity changed')
    environments={}
    for lane in QUEUES:
        w=out/'workers'/('gpu'+lane);pre=read_json(w/'preflight.json');environments[lane]=pre['environment']
        if pre['status']!='PASS' or pre['release']!=release or pre['prepared_sha256']!=sha(out/'prepared.json') or pre['queue']!=QUEUES[lane] or pre['physical_gpu']!=int(lane):raise RuntimeError('Worker preflight changed')
        for phase in ('preflight','fit','evaluate'):
            r=read_json(w/(phase+'_exit.json'))
            if r['exit']!=0 or r['phase']!=phase or r['physical_gpu']!=int(lane):raise RuntimeError('Worker failed or wrong lane')
    freeze=read_json(out/'freeze.json')
    if receipt['freeze_sha256']!=sha(out/'freeze.json') or freeze['study']!=STUDY or freeze['release']!=release or freeze['prepared_sha256']!=sha(out/'prepared.json') or freeze['worker_preflights']!=bindings or freeze['cross_card_parity_sha256']!=sha(out/'cross_card_parity.json'):raise RuntimeError('Evaluation freeze changed')
    for arm in ORDER:
        done=read_json(out/arm/'completed.json');cfg=read_json(out/arm/'config.json');lane=lane_for(arm)
        if done['step']!=5000 or done['config_hash']!=digest(cfg) or done['checkpoint_sha256']!=sha(out/arm/'last.pt') or freeze['models'].get(arm)!=done:raise RuntimeError('Final checkpoint binding changed: '+arm)
        if cfg['release']!=release or cfg['prepared_sha256']!=sha(out/'prepared.json') or cfg['preflight_sha256']!=bindings[lane] or cfg['physical_gpu']!=int(lane) or cfg['hardware_identity']!=environments[lane]['hardware_identity']:raise RuntimeError('Fit GPU/data/code identity changed: '+arm)
    summary=generate_report()
    if summary['status']!='COMPLETED' or len(summary['rows'])!=66 or summary['completed_fits']!=22:raise RuntimeError('All22 fits and66 rows required')
    manifest=read_json(ROOT/'release_manifest.json');members={n:ROOT/n for n in manifest['files']};members['release_manifest.json']=ROOT/'release_manifest.json'
    for name in ('prepared.json','source_plan.npy','source_manifest.json','cross_card_parity.json','freeze.json','summary.json','results.csv','report.md','pip_freeze.txt','run_completion.json','state.json','last_exit.txt','execution.log'):
        if (out/name).exists():members['evidence/'+name]=out/name
    for lane in QUEUES:
        for p in sorted((out/'workers'/('gpu'+lane)).glob('*')):
            if p.is_file() and p.suffix in ('.json','.log'):members['evidence/workers/gpu'+lane+'/'+p.name]=p
    for arm in ORDER:
        for p in sorted((out/arm).glob('*')):
            if p.is_file() and p.suffix in ('.json','.jsonl','.npz'):members[f'evidence/{arm}/{p.name}']=p
    members['data_preparation/import_record.json']=ROOT/'data_view/import_record.json'
    for name in ('data_config.json','runtime_python.txt'):
        if (ROOT/name).exists():members['data_preparation/'+name]=ROOT/name
    em={'kind':'ZXVAD_C04_DUALGPU_CODE_AND_ALL66_SCORES','campaign_status':'COMPLETED','release_identity':release,'summary_sha256':sha(out/'summary.json'),'files':{n:sha(p) for n,p in members.items()},'excluded':'Images/videos, checkpoint weights, credentials, private environments; hashes retained'}
    tmp=out/'review_bundle.zip.tmp'
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,strict_timestamps=False) as z:
        for name,p in sorted(members.items()):z.write(p,name)
        z.writestr('export_manifest.json',json.dumps(em,indent=2)+'\n')
    tmp.replace(out/'review_bundle.zip')
    write_json(out/'export_record.json',{'bundle_sha256':sha(out/'review_bundle.zip'),'summary_sha256':sha(out/'summary.json'),'files':len(members)})
    print('CPU review bundle:',out/'review_bundle.zip',flush=True)
if __name__=='__main__':export()
