"""Read existing lab images, make symlink view and V2 scored-label vectors."""
import argparse,hashlib,json,re,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from common import catalog,sha,digest,write_json,check_data,TARGETS

def trailing_id(name):
    m=re.search(r'(\d+)$',name)
    if not m:raise RuntimeError('Cannot map canonical test video: '+name)
    return int(m.group(1))

def link_folder(dst,src):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.is_symlink():
        if dst.resolve()!=src.resolve():raise RuntimeError('Existing symlink differs: '+str(dst))
    elif dst.exists():raise RuntimeError('Refusing to replace a real directory: '+str(dst))
    else:dst.symlink_to(src.resolve(),target_is_directory=True)

def make_labels(ref,videos):
    if ref['video_names'].tolist()!=[n for n,_ in videos]:raise RuntimeError('Reference video order mismatch')
    labels=[]
    for vi,(_,files) in enumerate(videos):
        use=ref['video']==vi
        if not np.array_equal(ref['frame'][use],np.arange(4,len(files))):raise RuntimeError('Reference frame count mismatch')
        y=np.full(len(files),-1,dtype=np.int8);y[4:]=ref['label'][use];labels.extend(y)
    return np.asarray(labels,dtype=np.int8)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--target-root',required=True);a=ap.parse_args()
    source=Path(a.source).resolve();target=Path(a.target_root).resolve();view=ROOT/'data_view'
    if source.parent.name.lower() not in ('training','train'):raise RuntimeError('Source is not a training split')
    expected=json.loads((ROOT/'reference/source_catalog.json').read_text())
    actual=catalog(source)
    if [{'video':n,'frames':len(f)} for n,f in actual]!=expected:raise RuntimeError('ShanghaiTech source video IDs/counts differ from audited V2; expected all330/274515')
    link_folder(view/'shanghai/training/frames',source)
    mappings={};config={'source_frames':str(view/'shanghai/training/frames'),'targets':{}}
    for t in TARGETS:
        frame_root=target/t/'testing/frames'
        actual=catalog(frame_root);mapping={}
        for name,files in actual:
            number=trailing_id(name)
            if number in mapping:raise RuntimeError('Ambiguous target video number: '+t)
            mapping[number]=(name,files)
        with np.load(ROOT/'reference'/('c04_'+t+'.npz'),allow_pickle=False) as ref:
            names=ref['video_names'].tolist()
            if set(mapping)!={trailing_id(n) for n in names}:raise RuntimeError('Target test video set differs: '+t)
            record=[]
            for name in names:
                old,files=mapping[trailing_id(name)]
                link_folder(view/t/'testing/frames'/name,frame_root/old)
                record.append({'canonical_video':name,'original_video':old,'original_directory':str((frame_root/old).resolve()),'frames':len(files)})
            videos=catalog(view/t/'testing/frames');labels=make_labels(ref,videos)
            label_path=view/t/'testing/scored_labels.npy'
            if label_path.exists():
                if not np.array_equal(np.load(label_path,allow_pickle=False),labels):raise RuntimeError('Existing scored labels differ')
            else:
                with label_path.open('wb') as f:np.save(f,labels,allow_pickle=False)
        mappings[t]=record;config['targets'][t]={'frames':str(view/t/'testing/frames'),'labels':str(label_path)}
    check_data(config)
    record={'kind':'AUDITED_V2_SCORED_LABEL_VIEW','source_all330':True,'source_frames':str(source),'source_total_frames':274515,'target_root':str(target),'images_copied':False,'images_modified':False,'scored_labels_source':'Fixed audited c04 at 3eb129ec5362ebd0575f9531d6ab178f4149e879','first4_labels':'Unknown/unscored: -1 sentinel; never used as ground truth','legacy_lab_labels_used':False,'reference_sha256':{t:sha(ROOT/'reference'/('c04_'+t+'.npz')) for t in TARGETS},'view_builder_sha256':sha(Path(__file__)),'target_mappings':mappings}
    p=view/'import_record.json'
    if p.exists() and json.loads(p.read_text())!=record:raise RuntimeError('Data import identity changed')
    write_json(p,record);write_json(ROOT/'data_config.json',config)
    print('All330 source / canonical V2 scored target labels / symlink view: PASS',flush=True)

if __name__=='__main__':main()
