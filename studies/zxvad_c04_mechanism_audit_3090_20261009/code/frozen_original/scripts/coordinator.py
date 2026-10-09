"""Parallel source checks/fits, global final-step freeze, parallel target scoring."""
import os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from common import read_json,write_json,sha
from schedule import QUEUES
OUT=ROOT/'outputs'
def status(phase,**extra):write_json(OUT/'state.json',{'phase':phase,'updated_unix':time.time(),**extra})
def parallel_phase(phase):
    status('BOTH_'+phase.upper())
    jobs=[]
    try:
        for lane in QUEUES:
            dst=OUT/'workers'/('gpu'+lane);dst.mkdir(parents=True,exist_ok=True)
            log=(dst/'execution.log').open('a',encoding='utf-8')
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=lane
            process=subprocess.Popen([sys.executable,str(ROOT/'scripts/worker.py'),'--lane',lane,'--phase',phase],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env)
            jobs.append((lane,process,log))
        # Let the healthy lane finish even if the other lane fails; preserve all progress.
        exits={lane:process.wait() for lane,process,_ in jobs}
    finally:
        for _,_,log in jobs:log.close()
    if any(exits.values()):raise RuntimeError(f'{phase} worker exits={exits}; inspect each worker log, then resume the same launch command')
    for lane in QUEUES:
        receipt=read_json(OUT/'workers'/('gpu'+lane)/(phase+'_exit.json'))
        if receipt['exit']!=0 or receipt['physical_gpu']!=int(lane) or receipt['phase']!=phase:raise RuntimeError('Invalid lane completion receipt')
    return exits
def cross_card_gate():
    records={k:read_json(OUT/'workers'/('gpu'+k)/'preflight.json') for k in QUEUES}
    binding={k:sha(OUT/'workers'/('gpu'+k)/'preflight.json') for k in QUEUES}
    for lane,r in records.items():
        if r['status']!='PASS' or r['prepared_sha256']!=sha(OUT/'prepared.json') or r['queue']!=QUEUES[lane] or r['physical_gpu']!=int(lane):raise RuntimeError('Invalid source preflight')
    same_initial=records['0']['shared_original_initial_hashes']==records['1']['shared_original_initial_hashes']
    same_update=records['0']['seed17_original_updater_parity']['state_sha256']==records['1']['seed17_original_updater_parity']['state_sha256']
    value={'status':'PASS' if same_initial and same_update else 'FAILED','shared_initial_tensors_exact':same_initial,'first_source_baseline_model_Adam_RNG_exact':same_update,'worker_preflights':binding,'scope':'Disposable first source batch only; no assertion of identical complete5000-step cross-GPU training trajectories'}
    write_json(OUT/'cross_card_parity.json',value)
    if value['status']!='PASS':raise RuntimeError('Cross-card source parity failed; no formal fits started. Inspect cross_card_parity.json and worker preflights')
def main():
    subprocess.run([sys.executable,str(ROOT/'src/pipeline.py'),'--phase','prepare'],check=True)
    pre=parallel_phase('preflight');cross_card_gate()
    fits=parallel_phase('fit')
    status('FREEZING_ALL22_FINAL_CHECKPOINTS')
    subprocess.run([sys.executable,str(ROOT/'src/pipeline.py'),'--phase','freeze'],check=True)
    evaluations=parallel_phase('evaluate')
    write_json(OUT/'run_completion.json',{'status':'ALL_GPU_WORK_COMPLETED','worker_exits':{'preflight':pre,'fit':fits,'evaluate':evaluations},'queues':QUEUES,'wall_time_limit':None,'freeze_sha256':sha(OUT/'freeze.json')})
    status('EXPORTING_RESULTS')
    subprocess.run([sys.executable,str(ROOT/'scripts/export_results.py')],check=True)
    summary=read_json(OUT/'summary.json')
    if summary['status']!='COMPLETED' or summary['completed_fits']!=22 or len(summary['rows'])!=66:raise RuntimeError('Incomplete evidence cannot count as completion')
    status('COMPLETED',completed_fits=22,AUROC_rows=66,positive_macro_candidates=summary['positive_macro_candidates'])
    print('All22 fits and66 AUROC rows completed; manual publication command is now available.',flush=True)
if __name__=='__main__':main()
