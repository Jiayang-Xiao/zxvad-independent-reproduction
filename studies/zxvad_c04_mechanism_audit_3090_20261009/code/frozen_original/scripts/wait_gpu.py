"""Read-only physical GPU idle polling; no elapsed-time cutoff or process signals."""
import argparse,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def wait(gpu):
    out=ROOT/'outputs/workers'/('gpu'+str(gpu));out.mkdir(parents=True,exist_ok=True)
    while True:
        raw=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
        used,util=[int(x.strip()) for x in raw.split(',')]
        processes=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
        if used<300 and util<=2 and not processes:
            print(f'Physical GPU{gpu} idle; proceeding.',flush=True);return
        msg=f'Physical GPU{gpu} occupied: {used}MiB,{util}%. Waiting without CUDA context.'
        p=out/'state.json';tmp=p.with_suffix('.json.tmp');tmp.write_text(json.dumps({'phase':'WAITING_FOR_GPU','updated_unix':time.time(),'message':msg})+'\n');tmp.replace(p)
        print(msg,flush=True);time.sleep(30)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,choices=[0,1],required=True);a=ap.parse_args();wait(a.gpu)
