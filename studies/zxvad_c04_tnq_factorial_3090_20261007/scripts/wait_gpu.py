"""Observe designated physical GPU; do not initialize CUDA or stop other jobs."""
import argparse,subprocess,time,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
# State writing does not import torch, so waiting cannot acquire GPU memory.
def status(message):
    p=ROOT/'outputs/state.json';tmp=p.with_suffix('.json.tmp')
    tmp.write_text(json.dumps({'phase':'WAITING_FOR_GPU','updated_unix':time.time(),'message':message})+'\n');tmp.replace(p)
ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,required=True);a=ap.parse_args()
while True:
    raw=subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
    used,util=[int(x.strip()) for x in raw.split(',')]
    # Idle display overhead is allowed, CUDA processes are not.
    processes=subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    if used<300 and util<=2 and not processes:break
    message=f'Physical GPU{a.gpu} occupied: {used}MiB, {util}% utilization. Waiting without CUDA context.'
    status(message);print(message,flush=True);time.sleep(30)
print(f'Physical GPU{a.gpu} idle; proceeding.',flush=True)
