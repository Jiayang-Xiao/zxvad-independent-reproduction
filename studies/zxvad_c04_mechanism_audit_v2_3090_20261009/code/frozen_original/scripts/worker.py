"""One lane supervisor, one process lock, one log; never writes global status."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
ap=argparse.ArgumentParser();ap.add_argument('--lane',choices=['0','1'],required=True);ap.add_argument('--phase',choices=['preflight','fit','evaluate'],required=True);a=ap.parse_args()
out=ROOT/'outputs/workers'/('gpu'+a.lane);out.mkdir(parents=True,exist_ok=True)
def write(name,value):
    p=out/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,indent=2)+'\n');t.replace(p)
lock=(out/'worker.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
(out/'worker.pid').write_text(str(os.getpid())+'\n')
os.environ['CUDA_VISIBLE_DEVICES']=a.lane
rc=1
try:
    print(f'GPU{a.lane}: starting phase {a.phase}',flush=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/wait_gpu.py'),'--gpu',a.lane],check=True)
    if a.phase=='preflight':subprocess.run([sys.executable,str(ROOT/'scripts/check_gpu.py'),a.lane],check=True)
    cmd=[sys.executable,str(ROOT/'src/pipeline.py'),'--phase',a.phase,'--lane',a.lane]
    rc=subprocess.run(cmd).returncode
except BaseException as e:
    print(f'GPU{a.lane} failed: {e}',flush=True)
finally:
    write(a.phase+'_exit.json',{'physical_gpu':int(a.lane),'phase':a.phase,'exit':rc,'finished_unix':time.time()})
    if rc:write('state.json',{'phase':'FAILED','failed_stage':a.phase,'exit':rc,'updated_unix':time.time()})
sys.exit(rc)
