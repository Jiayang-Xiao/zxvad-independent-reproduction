"""Predeclared source-only fits, then final-step target evaluation, on independent GPU queues."""
import argparse,gc,json,random,time,os
from pathlib import Path
import numpy as np
import torch
from common import ROOT,TARGETS,SourceClips,build_plan,catalog,check_data,check_runtime,digest,metadata,read_frame,read_json,release_identity,seed_everything,sha,write_json
from model import ORDER,RECIPES,SEEDS,parse,CONFIG,make_models,one_update,make_generator
from schedule import STUDY,QUEUES,lane_for
LANE=None
def lane_dir():
    assert LANE in QUEUES
    return OUT/"workers"/("gpu"+LANE)
def preflight_path():return lane_dir()/"preflight.json"
OUT=ROOT/'outputs'

def status(phase,**extra):
    dst=OUT if LANE is None else lane_dir()
    write_json(dst/'state.json',{'phase':phase,'updated_unix':time.time(),**extra})

def prepare():
    release=release_identity(); config=read_json(ROOT/'data_config.json')
    try:
        source,targets=check_data(config)
    except (FileNotFoundError,RuntimeError,NotADirectoryError,PermissionError) as e:
        status('WAITING_FOR_DATA',message=str(e),config=config)
        print('WAITING_FOR_DATA:',str(e),flush=True)
        return False
    view=read_json(ROOT/'data_view/import_record.json')
    source_meta=metadata(source)
    data={'source_frames':config['source_frames'],'source_metadata_sha256':digest(source_meta),'source_videos':len(source),'source_total_frames':sum(len(fs) for _,fs in source),'targets':{name:{'frame_root':config['targets'][name]['frames'],'label_path':config['targets'][name]['labels'],'metadata_sha256':digest(metadata(videos)),'labels_sha256':sha(config['targets'][name]['labels']),'videos':len(videos),'total_frames':len(y)} for name,(videos,y) in targets.items()}}
    plan=build_plan(source)
    plan_file=OUT/'source_plan.npy'
    if plan_file.exists():
        if not np.array_equal(plan,np.load(plan_file,allow_pickle=False)): raise RuntimeError('Source plan changed')
    else:
        with plan_file.open('wb') as f: np.save(f,plan,allow_pickle=False)
    value={'release':release,'view_record_sha256':sha(ROOT/'data_view/import_record.json'),'protocol_sha256':sha(ROOT/'protocol.json'),'data':data,'plan_sha256':sha(plan_file),'precision':'FP32; AMP/TF32 disabled','seeds':list(SEEDS),'sampling_plan_seed':17,'iterations':5000,'batch_size':8}
    path=OUT/'prepared.json'
    if path.exists() and read_json(path)!=value: raise RuntimeError('Prepared experiment identity changed; use a new output directory')
    write_json(path,value); write_json(OUT/'source_manifest.json',source_meta)
    print('Canonical source and target layout: PASS',flush=True)
    return True

def state_hash(model):
    import hashlib
    h=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def gpu_preflight():
    value=read_json(OUT/'prepared.json');path=preflight_path(); env=check_runtime()
    if path.exists():
        prior=read_json(path)
        if prior['prepared_sha256']==sha(OUT/'prepared.json') and prior['status']=='PASS' and prior['environment']==env:return
        raise RuntimeError('Preflight identity mismatch')
    status('SOURCE_PREFLIGHT')
    dataset=SourceClips(catalog(value['data']['source_frames']),np.load(OUT/'source_plan.npy',allow_pickle=False))
    clips,donors=zip(*(dataset[i] for i in range(8)))
    clip=torch.from_numpy(np.stack(clips)).cuda();donor=torch.from_numpy(np.stack(donors)).cuda()
    records={};shared={};diagnostic={}
    for arm in QUEUES[LANE]:
        recipe,seed=parse(arm);seed_everything(seed)
        models,opts,loss,aug=make_models(arm,'cuda:0')
        current={k:state_hash(m) for k,m in models.items() if k!='G'}
        current['G']=state_hash_filtered(models['G'])
        if seed in shared:assert current==shared[seed],'Shared initial tensors changed'
        else:shared[seed]=current
        with torch.no_grad():
            mask=models['O'](donor)
            diagnostic[arm]={'source_donor_mask_area':float(mask.mean()),'donors':8}
            if CONFIG[recipe]['support'] is not None:assert float(mask.mean())==CONFIG[recipe]['support']
        metrics=one_update(clip,donor,arm,0,models,opts,loss,aug,audit=True)
        torch.cuda.synchronize()
        assert metrics['n_bn_updates']==4
        assert all(not p.requires_grad and p.grad is None for p in models['O'].parameters())
        assert all(int(m.num_batches_tracked)==0 for m in models['O'].modules() if isinstance(m,torch.nn.BatchNorm2d))
        models['G'].eval()
        with torch.inference_mode():
            multi,_=models['G'](clip[:,:4].flatten(1,2));single,_=models['G'](clip[:1,:4].flatten(1,2))
            torch.testing.assert_close(multi[:1],single,rtol=1e-4,atol=1e-5)
        records[arm]={'losses':metrics,'G_parameters':sum(p.numel() for p in models['G'].parameters()),'inference_batch_parity_max_abs':float((multi[:1]-single).abs().max())}
        del models,opts,loss,aug,multi,single,mask;gc.collect();torch.cuda.empty_cache()
        print(arm,'source-only preflight PASS',flush=True)
    # Real GPU parity of direct original c04 versus the explicit-seed updater at17.
    from parity import check_parity
    parity=check_parity(clip,donor)
    write_json(path,{'status':'PASS','physical_gpu':int(LANE),'environment':env,'queue':QUEUES[LANE],'prepared_sha256':sha(OUT/'prepared.json'),'release':value['release'],'shared_original_initial_hashes':shared,'source_mask_diagnostic':diagnostic,'seed17_original_updater_parity':parity,'recipes':records,'scope':'Disposable source-only updates; discarded before fresh step0 research fits'})

def state_hash_filtered(model):
    import hashlib
    h=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        if key.startswith(('skip1.','skip2.','skip3.')):continue
        h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}

def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'].cpu()); torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])

def fit(candidate,workers):
    assert candidate in QUEUES[LANE]
    prepared=read_json(OUT/'prepared.json'); preflight=read_json(preflight_path())
    if preflight['status']!='PASS' or preflight['prepared_sha256']!=sha(OUT/'prepared.json'): raise RuntimeError('Invalid preflight')
    config={'study':STUDY,'candidate':candidate,'recipe':parse(candidate)[0],'prepared_sha256':sha(OUT/'prepared.json'),'preflight_sha256':sha(preflight_path()),'physical_gpu':int(LANE),'hardware_identity':preflight['environment']['hardware_identity'],'seed':parse(candidate)[1],'iterations':5000,'batch_size':8,'release':release_identity()}
    config_hash=digest(config); dst=OUT/candidate; dst.mkdir(exist_ok=True)
    if (dst/'config.json').exists() and read_json(dst/'config.json')!=config: raise RuntimeError('Fit identity changed: '+candidate)
    write_json(dst/'config.json',config)
    if (dst/'completed.json').exists():
        done=read_json(dst/'completed.json')
        if done['config_hash']!=config_hash or done['checkpoint_sha256']!=sha(dst/'last.pt'): raise RuntimeError('Completed fit changed')
        return True
    status('TRAINING',candidate=candidate,step=0,budget=5000)
    seed_everything(parse(candidate)[1]); models,opts,loss_fn,aug=make_models(candidate,'cuda:0')
    start,elapsed=0,0.
    if (dst/'last.pt').exists():
        ck=torch.load(dst/'last.pt',map_location='cuda:0',weights_only=False)
        if ck['config_hash']!=config_hash: raise RuntimeError('Checkpoint identity changed')
        for key,model in models.items(): model.load_state_dict(ck['models'][key])
        for key,opt in opts.items(): opt.load_state_dict(ck['optimizers'][key])
        restore_rng(ck['rng']); start,elapsed=ck['step'],ck['elapsed_seconds']; del ck
    source=catalog(prepared['data']['source_frames'])
    plan=np.load(OUT/'source_plan.npy',allow_pickle=False)
    loader=torch.utils.data.DataLoader(SourceClips(source,plan,start*8),batch_size=8,num_workers=workers,pin_memory=True,persistent_workers=workers>0,generator=torch.Generator().manual_seed(90417),**({'prefetch_factor':2} if workers>0 else {}))
    log=dst/'train.jsonl'
    if log.exists():
        keep=[]; damaged=False
        for line in log.read_text(encoding='utf-8').splitlines():
            try: entry=json.loads(line)
            except json.JSONDecodeError: damaged=True; continue
            if entry['step']<=start: keep.append(entry)
            else: damaged=True
        if damaged:
            log.replace(dst/f'uncommitted_train_{time.time_ns()}.jsonl')
            log.write_text(''.join(json.dumps(v)+'\n' for v in keep),encoding='utf-8')
    began=time.monotonic()
    def checkpoint(step):
        torch.cuda.synchronize()
        saved={'step':step,'config_hash':config_hash,'models':{k:m.state_dict() for k,m in models.items()},'optimizers':{k:o.state_dict() for k,o in opts.items()},'rng':rng_state(),'elapsed_seconds':elapsed+time.monotonic()-began}
        temp=dst/'last.pt.tmp'; torch.save(saved,temp); temp.replace(dst/'last.pt')
        write_json(dst/'state.json',{'phase':'TRAINED' if step==5000 else 'TRAINING','step':step,'checkpoint_step':step,'elapsed_seconds':saved['elapsed_seconds'],'config_hash':config_hash})
    print(f'Starting {candidate} at {start}/5000',flush=True)
    step=start
    if start<5000:
        for step,(clip,donor) in enumerate(loader,start+1):
            if step>5000: raise RuntimeError('Budget exceeded')
            clip,donor=clip.cuda(non_blocking=True),donor.cuda(non_blocking=True)
            entry=one_update(clip,donor,candidate,step-1,models,opts,loss_fn,aug)
            if entry['n_bn_updates']!=step*4: raise RuntimeError('Unexpected N BN update count')
            if step==start+1 or step%50==0:
                entry.update(step=step,elapsed_seconds=elapsed+time.monotonic()-began)
                with log.open('a',encoding='utf-8') as f: f.write(json.dumps(entry)+'\n')
                print(candidate,json.dumps(entry),flush=True)
                status('TRAINING',candidate=candidate,step=step,budget=5000)
            if step%250==0: checkpoint(step)
        if step!=5000: raise RuntimeError('Source plan ended before budget')
    checkpoint(5000)
    write_json(dst/'completed.json',{'phase':'TRAINED','step':5000,'config_hash':config_hash,'checkpoint_sha256':sha(dst/'last.pt')})
    del loader,models,opts,loss_fn,aug; gc.collect(); torch.cuda.empty_cache()
    return True

def freeze_evaluation(batch_size):
    prepared=read_json(OUT/'prepared.json'); config=read_json(ROOT/'data_config.json'); source,targets=check_data(config)
    if digest(metadata(source))!=prepared['data']['source_metadata_sha256']: raise RuntimeError('Source data changed during fitting')
    freeze={'study':STUDY,'author_confirmed':False,'release':release_identity(),'prepared_sha256':sha(OUT/'prepared.json'),'worker_preflights':{lane:sha(OUT/'workers'/('gpu'+lane)/'preflight.json') for lane in QUEUES},'cross_card_parity_sha256':sha(OUT/'cross_card_parity.json'),'models':{},'targets':{},'evaluation_batch':batch_size,'metric':'Per-video minmax negative PSNR; pooled frame AUROC; first4 excluded; final5000 only'}
    completed=list(ORDER)
    if not all((OUT/c/'completed.json').exists() for c in completed):raise RuntimeError('All22 final fits required before target evaluation')
    for candidate in completed:
        done=read_json(OUT/candidate/'completed.json'); cfg=read_json(OUT/candidate/'config.json')
        if done['step']!=5000 or done['checkpoint_sha256']!=sha(OUT/candidate/'last.pt') or done['config_hash']!=digest(cfg): raise RuntimeError('Incomplete fit: '+candidate)
        if cfg['prepared_sha256']!=freeze['prepared_sha256'] or cfg['preflight_sha256']!=freeze['worker_preflights'][lane_for(candidate)] or cfg['physical_gpu']!=int(lane_for(candidate)): raise RuntimeError('Fit preflight binding changed')
        freeze['models'][candidate]=done
    for name,(videos,y) in targets.items():
        old=prepared['data']['targets'][name]; metasha=digest(metadata(videos)); labelsha=sha(config['targets'][name]['labels'])
        if metasha!=old['metadata_sha256'] or labelsha!=old['labels_sha256']: raise RuntimeError('Target data changed: '+name)
        freeze['targets'][name]={'frame_metadata_sha256':metasha,'label_sha256':labelsha,'video_names':[v for v,_ in videos],'video_total_frames':[len(fs) for _,fs in videos]}
    path=OUT/'freeze.json'
    if path.exists() and read_json(path)!=freeze: raise RuntimeError('Evaluation freeze changed')
    write_json(path,freeze)

def evaluate(batch_size):
    freeze=read_json(OUT/'freeze.json');prepared=read_json(OUT/'prepared.json')
    if freeze['release']!=release_identity() or freeze['prepared_sha256']!=sha(OUT/'prepared.json') or freeze['worker_preflights'][LANE]!=sha(preflight_path()) or freeze['evaluation_batch']!=batch_size:raise RuntimeError('Evaluation identity changed')
    source,targets=check_data(read_json(ROOT/'data_config.json'))
    for name,(videos,y) in targets.items():
        if digest(metadata(videos))!=freeze['targets'][name]['frame_metadata_sha256'] or sha(read_json(ROOT/'data_config.json')['targets'][name]['labels'])!=freeze['targets'][name]['label_sha256']:raise RuntimeError('Target data changed')
    frozen_sha=sha(OUT/'freeze.json');seed_everything()
    for candidate in QUEUES[LANE]:
        if sha(OUT/candidate/'last.pt')!=freeze['models'][candidate]['checkpoint_sha256']:raise RuntimeError('Final checkpoint changed')
        model=make_generator(candidate,'cuda:0').eval()
        ck=torch.load(OUT/candidate/'last.pt',map_location='cpu',weights_only=False)
        if ck['step']!=5000 or ck['config_hash']!=freeze['models'][candidate]['config_hash']: raise RuntimeError('Loaded checkpoint mismatch')
        model.load_state_dict(ck['models']['G']); del ck
        for name,(videos,y) in targets.items():
            dst=OUT/candidate/(name+'.npz'); meta=dst.with_suffix('.json')
            if dst.exists() and meta.exists():
                old=read_json(meta)
                if old['freeze_sha256']==frozen_sha and old['scores_sha256']==sha(dst): continue
                raise RuntimeError('Existing scores changed')
            status('EVALUATING',candidate=candidate,target=name)
            raw,labels,video_ids,indices=[],[],[],[]; offset=0
            with torch.inference_mode():
                for vi,(video,files) in enumerate(videos):
                    window=[read_frame(p) for p in files[:4]]
                    xs,ys,ts=[],[],[]
                    def flush():
                        x=torch.from_numpy(np.stack(xs)).cuda(); target=torch.from_numpy(np.stack(ys)).cuda()
                        pred,_address=model(x)
                        mse=((pred-target)/2).square().mean((1,2,3))
                        raw.extend((10*torch.log10(mse.clamp_min(1e-12))).cpu().double().tolist())
                        labels.extend(int(y[offset+t]) for t in ts); video_ids.extend([vi]*len(ts)); indices.extend(ts)
                        xs.clear(); ys.clear(); ts.clear()
                    for t in range(4,len(files)):
                        observed=read_frame(files[t]); xs.append(np.stack(window).reshape(12,256,256)); ys.append(observed); ts.append(t)
                        window=window[1:]+[observed]
                        if len(xs)==batch_size: flush()
                    if xs: flush()
                    offset+=len(files)
                    print(candidate,name,video,len(raw),'frames',flush=True)
                    status('EVALUATING',candidate=candidate,target=name,scored_frames=len(raw),expected_frames=TARGETS[name][2])
            tmp=dst.with_suffix('.npz.tmp')
            with tmp.open('wb') as f: np.savez_compressed(f,raw=np.asarray(raw),label=np.asarray(labels),video=np.asarray(video_ids),frame=np.asarray(indices),video_names=np.asarray([n for n,_ in videos]))
            tmp.replace(dst); write_json(meta,{'freeze_sha256':frozen_sha,'scores_sha256':sha(dst),'scored_frames':len(raw)})
        del model; gc.collect(); torch.cuda.empty_cache()

def main():
    global LANE
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['prepare','preflight','fit','freeze','evaluate'],required=True);ap.add_argument('--lane',choices=list(QUEUES));ap.add_argument('--workers',type=int,default=4);ap.add_argument('--evaluation-batch',type=int,default=16);a=ap.parse_args()
    LANE=a.lane
    if a.phase in ('prepare','freeze'):
        if LANE is not None:raise RuntimeError('Coordinator phase cannot have a lane')
        if a.phase=='prepare':
            if not prepare():raise SystemExit(20)
        else:freeze_evaluation(a.evaluation_batch)
        return
    if LANE is None or os.environ.get('CUDA_VISIBLE_DEVICES')!=LANE:raise RuntimeError('Explicit assigned physical GPU required')
    env=check_runtime()
    if a.phase=='preflight':gpu_preflight();status('PREFLIGHT_PASSED');return
    pre=read_json(preflight_path())
    if pre['status']!='PASS' or pre['prepared_sha256']!=sha(OUT/'prepared.json') or pre['environment']!=env:raise RuntimeError('Worker preflight or runtime changed')
    if a.phase=='fit':
        parity=read_json(OUT/'cross_card_parity.json')
        if parity['status']!='PASS' or parity['worker_preflights']!={k:sha(OUT/'workers'/('gpu'+k)/'preflight.json') for k in QUEUES}:raise RuntimeError('Cross-card parity changed')
        for arm in QUEUES[LANE]:fit(arm,a.workers)
        status('ALL_LANE_FITS_COMPLETED',completed_fits=11)
    else:evaluate(a.evaluation_batch);status('ALL_LANE_EVALUATIONS_COMPLETED',AUROC_rows=33)

if __name__=='__main__':main()
