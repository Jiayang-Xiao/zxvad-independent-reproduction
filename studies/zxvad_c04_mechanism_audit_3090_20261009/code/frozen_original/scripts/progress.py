"""Read-only progress; stdlib only, no CUDA context."""
import json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'
def read(p):return json.loads(p.read_text()) if p.exists() else {}
def tail(p,n=6):
    if p.exists():
        with p.open('rb') as f:
            f.seek(0,2);size=f.tell();f.seek(max(0,size-16384));lines=f.read().decode('utf-8',errors='replace').splitlines()
        print('\n'.join(lines[-n:]))
def active(pidfile,needle):
    if not pidfile.exists():return False
    try:
        pid=int(pidfile.read_text());cmd=Path('/proc')/str(pid)/'cmdline'
        return cmd.exists() and needle.encode() in cmd.read_bytes()
    except (ValueError,FileNotFoundError,PermissionError):return False
print('Runner active' if active(OUT/'runner.pid',str(ROOT/'scripts/run.sh')) else 'Runner not active')
print(json.dumps(read(OUT/'state.json'),indent=2))
queues=read(ROOT/'protocol.json')['physical_gpu_queues']
for lane,arms in queues.items():
    w=OUT/'workers'/('gpu'+lane)
    print('\n=== Physical GPU'+lane+' ===')
    print('Worker active' if active(w/'worker.pid',str(ROOT/'scripts/worker.py')) else 'Worker inactive')
    print(json.dumps(read(w/'state.json'),indent=2))
    print('candidate,phase,checkpoint_step,completed_targets')
    for arm in arms:
        d=OUT/arm;state=read(d/'state.json')
        targets=sum((d/(t+'.npz')).exists() and (d/(t+'.json')).exists() for t in ('ped1','ped2','avenue'))
        print(f"{arm},{state.get('phase','NOT_STARTED')},{state.get('checkpoint_step',0)},{targets}/3")
    print('Latest worker log:');tail(w/'execution.log')
print('\n=== Coordinator log ===');tail(OUT/'execution.log',10)
print('Last exit (historical on resume):', (OUT/'last_exit.txt').read_text().strip() if (OUT/'last_exit.txt').exists() else 'not yet written')
summary=read(OUT/'summary.json')
if summary:
    print('AUROC rows:',len(summary.get('rows',[])))
    print('Macro AUROC:',summary.get('macro_AUROC'))
    print('Positive macro candidates:',summary.get('positive_macro_candidates'))
