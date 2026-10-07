"""Focused read-only inventory, including raw videos, archives and symlink targets."""
import argparse,json,os,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--dataset-root',default='/home/featurize/datasets');args=ap.parse_args()
locations=[Path(args.dataset_root),Path('/home/featurize/work'),root/'data']
result={'roots':[],'entries':[],'label_or_video_candidates':[],'bounded':False};deadline=time.monotonic()+30
seen=set()
for base in locations:
    if not base.exists():continue
    result['roots'].append(str(base))
    queue=[(base,0)]
    while queue:
        p,depth=queue.pop(0)
        if time.monotonic()>deadline or len(seen)>3000:result['bounded']=True;break
        resolved=str(p.resolve())
        if resolved in seen:continue
        seen.add(resolved)
        try:items=sorted(p.iterdir())
        except (PermissionError,OSError):continue
        directories=[q.name for q in items if q.is_dir()]
        files=[q for q in items if q.is_file()]
        if depth<=3 and len(result['entries'])<160:
            result['entries'].append({'path':str(p),'resolved':resolved,'symlink':p.is_symlink(),'subdirectories':directories[:12],'subdirectory_count':len(directories),'files':[{'name':q.name,'bytes':q.stat().st_size} for q in files[:8]],'file_count':len(files)})
        for q in files:
            suffix=q.suffix.lower()
            if suffix in {'.avi','.mp4','.zip','.tar','.gz','.rar','.7z','.mat'} or (suffix in {'.npy','.npz','.txt'} and any(t in q.name.lower() for t in ['label','ground','mask','gt'])):
                if len(result['label_or_video_candidates'])<120:result['label_or_video_candidates'].append({'path':str(q),'bytes':q.stat().st_size})
        if p.name.lower()=='frames' or depth>=5:continue
        for q in items:
            if q.is_dir() and not q.name.startswith('.') and q.name not in {'envs','venv','node_modules','site-packages','outputs','experiments'}:queue.append((q,depth+1))
    if result['bounded']:break
print(json.dumps(result,ensure_ascii=False,indent=2))
