"""Create a V2 view of existing canonical data, preserving clip-label alignment.

Reads images only as file metadata; no copy/decode/training. Run before prepared.json.
The old21-file V2 model/evaluation release is not modified.
"""
import argparse,hashlib,io,json,os
from pathlib import Path
import numpy as np

RELEASE_MANIFEST_SHA='fd4a06f79fb2e73173d5a934d26a53de7bb4d35b6265b9df89b2834e1019ac67'
EXPECTED={'shanghaitech':(330,274515),'ped1':(36,7200),'ped2':(12,2010),'avenue':(21,15324)}
EXTENSIONS={'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def plan_view(canonical,workspace,expected=EXPECTED):
    canonical=Path(canonical).resolve();workspace=Path(workspace).resolve();view=workspace/'data_view'
    links=[];arrays={};indices={};records={}
    for dataset,(nv,total) in expected.items():
        base=canonical/dataset;index=base/'index.json';doc=json.loads(index.read_text(encoding='utf-8'))
        split='train' if dataset=='shanghaitech' else 'test'
        rows=sorted(doc[split],key=lambda c:str(c['clip_id']))
        if len(rows)!=nv or sum(int(c['n_frames']) for c in rows)!=total:raise RuntimeError('Canonical count mismatch: '+dataset)
        all_ids=[str(c['clip_id']) for s in ('train','test') for c in doc.get(s,[])]
        if len(all_ids)!=len(set(all_ids)):raise RuntimeError('Train/test clip IDs overlap: '+dataset)
        all_paths={}
        for s in ('train','test'):
            for c in doc.get(s,[]):
                p=Path(c['frames_dir']);p=(p if p.is_absolute() else base/p).resolve()
                if p in all_paths:raise RuntimeError('Train/test frame directories overlap: '+dataset)
                all_paths[p]=s
        labels=[];names=[]
        destination=view/dataset/('training' if split=='train' else 'testing')/'frames'
        for c in rows:
            name=str(c['clip_id'])
            if name in {'.','..'} or Path(name).name!=name or '/' in name or '\\' in name:raise RuntimeError('Unsafe clip name')
            p=Path(c['frames_dir']);p=(p if p.is_absolute() else base/p).resolve()
            if canonical not in p.parents:raise RuntimeError('Frame path outside canonical tree: '+str(p))
            if not p.is_dir():raise RuntimeError('Missing frames: '+str(p))
            fs=sorted(f for f in p.iterdir() if f.is_file() and f.suffix.lower() in EXTENSIONS)
            n=int(c['n_frames'])
            if len(fs)!=n or n<5:raise RuntimeError('Frame count mismatch: '+name)
            if [f.stem for f in fs]!=[f'{i:06d}' for i in range(n)]:raise RuntimeError('Expected canonical0-based contiguous frame names: '+name)
            if split=='test':
                y=np.asarray(c.get('labels',[]))
                if y.shape!=(n,) or not np.isin(y,[0,1]).all():raise RuntimeError('Invalid labels: '+name)
                labels.append(y.astype(np.uint8))
            names.append(name);links.append((destination/name,p))
        if split=='test':
            y=np.concatenate(labels)
            if len(np.unique(y))!=2:raise RuntimeError('Target requires both classes')
            buff=io.BytesIO();np.save(buff,y,allow_pickle=False);arrays[view/f'frame_labels_{dataset}.npy']=buff.getvalue()
        indices[dataset]=sha(index)
        records[dataset]={'split':split,'videos':nv,'frames':total,'sorted_clip_ids':names,'index':str(index),'label_source':doc.get('source_notes',doc.get('label_source'))}
    config={'source_frames':str(view/'shanghaitech'/'training'/'frames'),'targets':{t:{'frames':str(view/t/'testing'/'frames'),'labels':str(view/f'frame_labels_{t}.npy')} for t in ('ped1','ped2','avenue')}}
    record={'identity':'ZXVAD_V2_CANONICAL_VIEW_20261007','canonical_root':str(canonical),'index_sha256':indices,'datasets':records,'view_label_sha256':{p.name:hashlib.sha256(b).hexdigest() for p,b in arrays.items()},'view_builder_sha256':sha(__file__),'images_copied':False,'source_heldout_scenes':[],'source_all330':expected['shanghaitech'][0]==330,'Avenue_label_version':'Canonical index labels, derived by handoff from ground_truth_demo volLabel.any; differs on429 scored frames from previous legacy labels. Do not merge historical identities.'}
    return links,arrays,config,record

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--canonical',default='/home/featurize/datasets/canonical');ap.add_argument('--workspace',default='/home/featurize/work/zxvad-independent-reproduction-v2');args=ap.parse_args()
    workspace=Path(args.workspace).resolve()
    if sha(workspace/'release_manifest.json')!=RELEASE_MANIFEST_SHA:raise SystemExit('Original V2 release required; run its bootstrap first')
    import fcntl
    lock=(workspace/'outputs'/'run.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('V2 installer/runner is active. Wait for WAITING_FOR_DATA and retry.')
    if (workspace/'outputs'/'prepared.json').exists():raise SystemExit('Experiment already frozen; do not change data identity')
    links,arrays,config,record=plan_view(args.canonical,workspace)
    manifest=workspace/'data_view'/'import_record.json'
    if manifest.exists() and json.loads(manifest.read_text())!=record:raise SystemExit('Existing canonical view identity differs')
    # Validate all existing outputs before writing any link, label or configuration.
    allowed={}
    for dst,src in links:
        allowed.setdefault(dst.parent,set()).add(dst.name)
        if os.path.lexists(dst) and (not dst.is_symlink() or dst.resolve()!=src):raise SystemExit('Existing view path differs: '+str(dst))
    for parent,names in allowed.items():
        if parent.exists() and set(p.name for p in parent.iterdir())-names:raise SystemExit('Unexpected videos in view: '+str(parent))
    for dst,data in arrays.items():
        if dst.exists() and dst.read_bytes()!=data:raise SystemExit('Existing generated labels differ: '+str(dst))
    for dst,src in links:
        dst.parent.mkdir(parents=True,exist_ok=True)
        if not os.path.lexists(dst):dst.symlink_to(src,target_is_directory=True)
    for dst,data in arrays.items():
        if not dst.exists():dst.write_bytes(data)
    def write(path,value):
        temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');temp.replace(path)
    write(manifest,record);write(workspace/'data_config.json',config)
    print('Canonical view prepared:330 normal source videos;69 target videos; matching sorted flat labels; images referenced by symlink.')
    print('Data provenance:',manifest)
    print('V2 launch:',workspace/'scripts'/'launch.sh')

if __name__=='__main__':main()
