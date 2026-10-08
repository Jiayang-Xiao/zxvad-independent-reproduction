"""Fixed protocol, snapshot identities and deterministic source-only data plans."""
import hashlib,json,os,random,time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
TARGETS = {'ped1':(36,7200,7056),'ped2':(12,2010,1962),'avenue':(21,15324,15240)}
EXTENSIONS = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1048576),b''): h.update(part)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def write_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8'); temp.replace(path)

def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def release_identity():
    manifest=read_json(ROOT/'release_manifest.json')
    for name,expected in manifest['files'].items():
        if sha(ROOT/name)!=expected: raise RuntimeError('Release modified: '+name)
    return digest(manifest)

def catalog(root):
    root=Path(root)
    return [(v.name,sorted(p for p in v.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)) for v in sorted(root.iterdir()) if v.is_dir() and not v.name.startswith('.')]

def metadata(videos):
    return [{'video':name,'frames':[[p.name,p.stat().st_size,p.stat().st_mtime_ns] for p in files]} for name,files in videos]

def read_frame(path):
    import cv2
    cv2.setNumThreads(0)
    image=cv2.imread(str(path),cv2.IMREAD_COLOR)
    if image is None: raise RuntimeError('Cannot decode: '+str(path))
    image=cv2.cvtColor(cv2.resize(image,(256,256),interpolation=cv2.INTER_LINEAR),cv2.COLOR_BGR2RGB)
    return ((image.astype(np.float32)/127.5)-1).transpose(2,0,1).copy()

def seed_everything(seed=17):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)

def build_plan(videos,steps=5000):
    # Reference-style round-robin source generator and a declared NumPy shuffle buffer.
    starts=np.random.RandomState(17); shuffle=np.random.RandomState(104746); donors=np.random.RandomState(390)
    cursor=0
    def draw():
        nonlocal cursor
        v=cursor%len(videos); cursor+=1
        return v,int(starts.randint(len(videos[v][1])-4))
    buffer=[draw() for _ in range(1000)]
    plan=np.empty((steps*8,4),dtype=np.int32)
    for i in range(steps*8):
        pos=int(shuffle.randint(1000)); v,s=buffer[pos]; buffer[pos]=draw()
        dv=int(donors.randint(len(videos))); ds=int(donors.randint(len(videos[dv][1])))
        plan[i]=[v,s,dv,ds]
    return plan

class SourceClips(torch.utils.data.Dataset):
    def __init__(self,videos,plan,start=0): self.videos,self.plan,self.start=videos,plan,start
    def __len__(self): return len(self.plan)-self.start
    def __getitem__(self,index):
        v,s,dv,ds=self.plan[self.start+index]
        frames=self.videos[int(v)][1]
        return np.stack([read_frame(p) for p in frames[int(s):int(s)+5]]),read_frame(self.videos[int(dv)][1][int(ds)])

def environment():
    import cv2,kornia,torchvision,pytorch_metric_learning,platform
    return {'python':platform.python_version(),'torch':torch.__version__,'torchvision':torchvision.__version__,'cuda_build':torch.version.cuda,'kornia':kornia.__version__,'pml':pytorch_metric_learning.__version__,'numpy':np.__version__,'opencv':cv2.__version__,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'compiled_arch':torch.cuda.get_arch_list()}

def check_runtime():
    env=environment()
    required={'torch':'2.7.1','torchvision':'0.22.1','cuda_build':'12.8','kornia':'0.6.9','pml':'1.6.3','numpy':'1.26.4','opencv':'4.10.0'}
    for key,value in required.items():
        actual=str(env[key]).split('+')[0]
        if actual!=value: raise RuntimeError(f'Wrong runtime {key}: {actual} != {value}')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    if torch.cuda.get_device_capability(0)==(12,0) and 'sm_120' not in env['compiled_arch']: raise RuntimeError('Missing sm_120 binary support')
    return env

def check_data(config):
    source_root=Path(config['source_frames']).resolve()
    if source_root.parent.name.lower() not in {'training','train'}: raise RuntimeError('Source must be a training/frames directory, never a testing split')
    source=catalog(config['source_frames'])
    if len(source)!=330 or any(len(fs)<5 for _,fs in source): raise RuntimeError('Source requires 330 usable ShanghaiTech NORMAL training videos')
    targets={}
    for name,(nv,total,scored) in TARGETS.items():
        item=config['targets'][name]; target_root=Path(item['frames']).resolve()
        if source_root==target_root or source_root in target_root.parents or target_root in source_root.parents: raise RuntimeError('Source and target paths overlap')
        videos=catalog(item['frames']); y=np.load(item['labels'],allow_pickle=False).reshape(-1)
        if len(videos)!=nv or sum(len(fs) for _,fs in videos)!=total or sum(len(fs)-4 for _,fs in videos)!=scored or len(y)!=total: raise RuntimeError('Canonical frames/labels mismatch: '+name)
        if not np.isin(y,[0,1]).all() or len(np.unique(y))!=2: raise RuntimeError('Invalid target labels: '+name)
        targets[name]=(videos,y)
    return source,targets
