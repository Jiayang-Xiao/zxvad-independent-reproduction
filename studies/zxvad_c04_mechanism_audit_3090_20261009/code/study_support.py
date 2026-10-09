"""Frozen parent identities for the32-fit source-heldout mechanism audit (no CUDA work)."""
import hashlib,json,shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'
REFERENCE=ROOT/'parent_reference'
from spec import QUEUES, STUDY
STUDY='ZXVAD_C04_MECHANISM_AUDIT_3090_20261009'

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1048576),b''):h.update(part)
    return h.hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    content=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(content,encoding='utf-8');tmp.replace(path)
def require(condition,message):
    if not condition:raise RuntimeError(message)
def copy_exact(source,dest,expected):
    require(source.is_file() and sha(source)==expected,'Frozen input changed/missing: '+str(source))
    if dest.exists():require(sha(dest)==expected,'Existing copied input changed: '+str(dest))
    else:dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)

def original_path(parent,name):
    if name.startswith('evidence/'):return parent/'outputs'/name[len('evidence/'):]
    if name=='data_preparation/import_record.json':return parent/'data_view/import_record.json'
    if name in ('data_preparation/data_config.json','data_preparation/runtime_python.txt'):return parent/Path(name).name
    raise RuntimeError('Unexpected parent input path: '+name)

def contract():return read(ROOT/'parent_contract.json')

def verify_original_code():
    c=contract();parent=Path(c['parent_root']);p=ROOT/'frozen_original/release_manifest.json'
    require(sha(p)==c['parent_release_manifest_sha256'],'Vendored original manifest changed')
    m=read(p);require(digest(m)==c['parent_release_identity'] and len(m['files'])==41,'Original release identity changed')
    for name,h in m['files'].items():
        require(sha(ROOT/'frozen_original'/name)==h,'Vendored original code changed: '+name)
        if parent.exists():require(sha(parent/name)==h,'Live parent code changed: '+name)
    return m

def verify_existing_parent_weights():
    """Old weights are never loaded. Verify them if retained; score evidence is pinned independently."""
    c=contract();parent=Path(c['parent_root']);records={}
    for arm in c['historical_gpu']:
        expected=read(REFERENCE/f'evidence/{arm}/completed.json')['checkpoint_sha256']
        p=parent/'outputs'/arm/'last.pt';exists=p.is_file()
        if exists:require(sha(p)==expected,'Retained parent checkpoint changed: '+arm)
        records[arm]={'checkpoint_sha256_recorded':expected,'file_present_and_verified':exists,'weights_used':False}
    return records

def expected_binding(availability):
    c=contract()
    return {'study':STUDY,'parent_commit':c['parent_commit'],'parent_results_commit':c['parent_results_commit'],'contract_sha256':sha(ROOT/'parent_contract.json'),'parent_export_manifest_sha256':c['parent_export_manifest_sha256'],'parent_reference_files':c['files'],'parent_release':c['parent_release_identity'],'queues':QUEUES,'parent_weights_used':False,'parent_checkpoint_availability_at_binding':availability,'historic_inputs_frozen_before_new_fits':True,'source_image_identity':'Exact original paths and metadata; historic image bytes were not separately hashed'}

def prepare_parent():
    c=contract();parent=Path(c['parent_root']);OUT.mkdir(exist_ok=True)
    require(read(parent/'outputs/state.json')['phase']=='COMPLETED','Parent screening is not completed')
    require((parent/'outputs/last_exit.txt').read_text().strip()=='0','Parent runner did not exit successfully')
    require(sha(parent/'release_manifest.json')==c['parent_release_manifest_sha256'],'Parent release manifest changed')
    m=read(parent/'release_manifest.json')
    require(digest(m)==c['parent_release_identity'],'Parent code release changed')
    require(len(m['files'])==41,'Expected41 original frozen files')
    for name,h in m['files'].items():
        require(sha(parent/name)==h and sha(ROOT/'frozen_original'/name)==h,'Original frozen code differs: '+name)
    require(sha(ROOT/'frozen_original/release_manifest.json')==c['parent_release_manifest_sha256'],'Vendored original manifest differs')
    require(sha(ROOT/'parent_export_manifest.json')==c['parent_export_manifest_sha256'],'Pinned parent export manifest changed')
    em=read(ROOT/'parent_export_manifest.json')
    for name,h in c['files'].items():
        require(em['files'].get(name)==h,'Parent input not bound to original export: '+name)
        copy_exact(original_path(parent,name),REFERENCE/name,h)
    for name in ('data_config.json','runtime_python.txt'):
        copy_exact(parent/name,ROOT/name,c['files']['data_preparation/'+name])
    copy_exact(parent/'data_view/import_record.json',ROOT/'data_view/import_record.json',c['files']['data_preparation/import_record.json'])
    dst=OUT/'parent_binding.json'
    current_weights=verify_existing_parent_weights()
    initial_weights=read(dst)['parent_checkpoint_availability_at_binding'] if dst.exists() else current_weights
    binding=expected_binding(initial_weights)
    if dst.exists():require(read(dst)==binding,'Parent binding changed; refuse to mix runs')
    else:write(dst,binding)
    validate_parent()
    print('Pinned parent code, data identities and physical-card environments: PASS',flush=True)

def validate_parent():
    c=contract();b=read(OUT/'parent_binding.json')
    verify_original_code()
    availability=b.get('parent_checkpoint_availability_at_binding',{})
    require(set(availability)==set(c['historical_gpu']),'Missing/extra parent checkpoint availability')
    require(b==expected_binding(availability),'Parent binding fields changed')
    parity=OUT/'cross_card_parity.json'
    if parity.exists():require(read(parity).get('parent_binding_sha256')==sha(OUT/'parent_binding.json'),'Parent binding changed after source parity gate')
    require(b['study']==STUDY and b['parent_commit']==c['parent_commit'] and b['contract_sha256']==sha(ROOT/'parent_contract.json'),'Parent binding mismatch')
    require(b['parent_reference_files']==c['files'] and b['queues']==QUEUES and b['parent_weights_used'] is False,'Parent binding changed')
    require(sha(ROOT/'parent_export_manifest.json')==c['parent_export_manifest_sha256'],'Pinned parent export manifest changed')
    for name,h in c['files'].items():require(sha(REFERENCE/name)==h,'Frozen historical input changed: '+name)
    for name in ('data_config.json','runtime_python.txt'):
        require(sha(ROOT/name)==c['files']['data_preparation/'+name],'Current runtime/data configuration differs')
    require(sha(ROOT/'data_view/import_record.json')==c['files']['data_preparation/import_record.json'],'Current data view differs')
    p=read(REFERENCE/'evidence/prepared.json');f=read(REFERENCE/'evidence/freeze.json')
    require(p['release']==f['release']==c['parent_release_identity'],'Parent release binding mismatch')
    require(f['prepared_sha256']==sha(REFERENCE/'evidence/prepared.json') and f['evaluation_batch']==16,'Parent evaluator/prepared identity differs')
    require(p['plan_sha256']==sha(REFERENCE/'evidence/source_plan.npy'),'Parent plan binding mismatch')
    for lane in QUEUES:
        pre=read(REFERENCE/f'evidence/workers/gpu{lane}/preflight.json')
        require(pre['environment']==c['physical_gpu_environment'][lane] and pre['status']=='PASS','Parent environment differs')
        require(f['worker_preflights'][lane]==sha(REFERENCE/f'evidence/workers/gpu{lane}/preflight.json'),'Parent preflight binding mismatch')
    for arm,lane in c['historical_gpu'].items():
        cfg=read(REFERENCE/f'evidence/{arm}/config.json');done=read(REFERENCE/f'evidence/{arm}/completed.json')
        require(cfg['physical_gpu']==lane and cfg['seed']==17 and cfg['iterations']==5000 and cfg['batch_size']==8,'Parent fit setting differs')
        require(done['step']==5000 and done['config_hash']==digest(cfg) and f['models'][arm]==done,'Parent final checkpoint records differ')
        for target in ('ped1','ped2','avenue'):
            meta=read(REFERENCE/f'evidence/{arm}/{target}.json')
            require(meta['freeze_sha256']==sha(REFERENCE/'evidence/freeze.json') and meta['scores_sha256']==sha(REFERENCE/f'evidence/{arm}/{target}.npz'),'Parent scored NPZ binding differs')
    weights=verify_existing_parent_weights()
    for arm,item in b['parent_checkpoint_availability_at_binding'].items():
        require(set(item)=={'checkpoint_sha256_recorded','file_present_and_verified','weights_used'} and type(item['file_present_and_verified']) is bool,'Invalid parent availability schema')
        require(item['checkpoint_sha256_recorded']==weights[arm]['checkpoint_sha256_recorded'] and item['weights_used'] is False,'Parent checkpoint declaration changed')
    return b

def validate_prepared():
    validate_parent();p=read(OUT/'prepared.json');old=read(REFERENCE/'evidence/prepared.json')
    for k in ('data','view_record_sha256','precision','seeds','sampling_plan_seed','iterations','batch_size'):
        require(p[k]==old[k],'New prepared input differs from parent: '+k)
    require(p['plan_sha256']==sha(OUT/'source_plan.npy')==sha(ROOT/'reference/source_train_plan.npy'),'Frozen297-video training plan differs')
    for name in ('source_split.json','source_probe_plan.npy'):
        require(sha(OUT/name)==sha(ROOT/'reference'/name),'Source validation plan/split changed')
    require(p['protocol_sha256']==sha(ROOT/'protocol.json') and p['release']==digest(read(ROOT/'release_manifest.json')),'New code/protocol binding differs')
    import numpy as np
    split=read(OUT/'source_split.json');train=set(split['training_indices']);valid=set(split['validation_indices'])
    require(len(train)==297 and len(valid)==33 and not train&valid and train|valid==set(range(330)),'Source videos must partition297/33')
    plan=np.load(OUT/'source_plan.npy',allow_pickle=False);probe=np.load(OUT/'source_probe_plan.npy',allow_pickle=False)
    require(plan.shape==(40000,4) and probe.shape==(256,4) and set(plan[:,0])<=train and set(plan[:,2])<=train and set(probe[:,0])<=valid and set(probe[:,2])<=valid,'Recipient/donor source leakage detected')
    return p

def validate_preflight(lane):
    lane=str(lane);validate_prepared()
    new=read(OUT/f'workers/gpu{lane}/preflight.json');old=read(REFERENCE/f'evidence/workers/gpu{lane}/preflight.json')
    require(new['queue']==QUEUES[lane] and new['physical_gpu']==int(lane) and new['status']=='PASS','New lane preflight differs')
    require(new['environment']==old['environment'],'Same physical GPU/runtime environment required for comparison')
    require(new['shared_original_initial_hashes']==old['shared_original_initial_hashes'],'Shared initialization changed')
    require(new['seed17_original_updater_parity']['status']=='PASS','Current source direct-c04 updater parity failed')
    require(new['prepared_sha256']==sha(OUT/'prepared.json'),'New preflight prepared binding differs')
    return new
