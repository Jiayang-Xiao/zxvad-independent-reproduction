"""Source split, single-family c04 updates and frozen diagnostics."""
import argparse,importlib,os,sys
from pathlib import Path
import spec
import study_support as support
ROOT=Path(__file__).resolve().parent
STUDY,QUEUES,ORDER=spec.STUDY,spec.QUEUES,spec.ORDER
PHASES=('prepare','preflight','fit','probe','selection','freeze','evaluate')
_PIPELINE=None

def verify_release_files():
    manifest=support.read(ROOT/'release_manifest.json')
    support.require(manifest['study']==STUDY,'Wrong release study')
    for name,expected in manifest['files'].items():
        support.require(support.sha(ROOT/name)==expected,'Release modified: '+name)

def original_pipeline():
    global _PIPELINE
    if _PIPELINE is not None:return _PIPELINE
    source=(ROOT/'frozen_original/src').resolve();sys.path.insert(0,str(source))
    common=importlib.import_module('common')
    support.require(Path(common.__file__).resolve().parent==source,'Wrong common module')
    common.ROOT=ROOT
    def frozen_plan(videos,steps=5000):
        import numpy as np
        expected=support.read(ROOT/'reference/source_catalog.json')
        actual=[{'video':name,'frames':len(files)} for name,files in videos]
        support.require(actual==expected and steps==5000,'Source catalogue/step budget changed')
        return np.load(ROOT/'reference/source_train_plan.npy',allow_pickle=False).copy()
    common.build_plan=frozen_plan
    # Original schedule asserts its original22-arm order before adaptation.
    schedule=importlib.import_module('schedule');model=importlib.import_module('model')
    support.require(Path(schedule.__file__).resolve().parent==source and Path(model.__file__).resolve().parent==source,'Wrong model/schedule')
    schedule.STUDY,schedule.QUEUES=STUDY,QUEUES
    import experiment_update as update
    model.CONFIG={recipe:{**model.DEFAULT,**settings} for recipe,settings in spec.CONFIG.items()}
    model.CONFIG['B']=dict(model.DEFAULT)
    model.ORDER,model.RECIPES,model.SEEDS,model.parse=ORDER,tuple(spec.CONFIG),spec.SEEDS,spec.parse
    model.make_models,model.make_generator,model.one_update=update.make_models,update.make_generator,update.one_update
    p=importlib.import_module('pipeline')
    support.require(Path(p.__file__).resolve().parent==source,'Wrong pipeline module')
    support.require(p.ROOT==ROOT and p.OUT==ROOT/'outputs' and p.QUEUES==QUEUES and p.ORDER==ORDER and p.STUDY==STUDY,'Adapter binding failed')
    _PIPELINE=p;return p

def cross_card_receipt(p):
    value=p.read_json(p.OUT/'cross_card_parity.json')
    expected={lane:p.sha(p.OUT/'workers'/('gpu'+lane)/'preflight.json') for lane in QUEUES}
    support.require(value.get('status')=='PASS' and value.get('worker_preflights')==expected
                    and value.get('shared_initial_tensors_exact') is True
                    and value.get('first_source_baseline_model_Adam_RNG_exact') is True
                    and value.get('parent_binding_sha256')==p.sha(p.OUT/'parent_binding.json'),'Source parity receipt changed')

def validate_live_data(p):
    prepared=p.read_json(p.OUT/'prepared.json');config=p.read_json(ROOT/'data_config.json')
    source,targets=p.check_data(config)
    support.require(p.digest(p.metadata(source))==prepared['data']['source_metadata_sha256'],'Source metadata changed')
    for name,(videos,labels) in targets.items():
        prior=prepared['data']['targets'][name]
        support.require(p.digest(p.metadata(videos))==prior['metadata_sha256'] and p.sha(config['targets'][name]['labels'])==prior['labels_sha256'],'Target identity changed: '+name)
    return source,targets

def freeze_evaluation(p,batch_size):
    import source_selection
    source_selection.validate_source_validation()
    selection=p.read_json(p.OUT/'source_selection.json')
    support.require(selection['source_validation_sha256']==p.sha(p.OUT/'source_validation.json'),'Source choice changed')
    source,targets=validate_live_data(p);config=p.read_json(ROOT/'data_config.json')
    value={'study':STUDY,'author_confirmed':False,'release':p.release_identity(),
           'prepared_sha256':p.sha(p.OUT/'prepared.json'),
           'worker_preflights':{lane:p.sha(p.OUT/'workers'/('gpu'+lane)/'preflight.json') for lane in QUEUES},
           'cross_card_parity_sha256':p.sha(p.OUT/'cross_card_parity.json'),
           'source_validation_sha256':p.sha(p.OUT/'source_validation.json'),
           'source_selection_sha256':p.sha(p.OUT/'source_selection.json'),
           'training_source_videos':297,'validation_source_videos':33,
           'models':{},'targets':{},'evaluation_batch':batch_size,
           'metric':'Primary per-video minmax negative PSNR pooled frame AUROC; first4 excluded; final5000 only',
           'diagnostic_readouts':list(spec.READOUTS)}
    for candidate in ORDER:
        done=p.read_json(p.OUT/candidate/'completed.json');cfg=p.read_json(p.OUT/candidate/'config.json');lane=spec.lane_for(candidate)
        support.require(done['step']==5000 and done['checkpoint_sha256']==p.sha(p.OUT/candidate/'last.pt') and done['config_hash']==p.digest(cfg),'Incomplete fit: '+candidate)
        support.require(cfg['prepared_sha256']==value['prepared_sha256'] and cfg['preflight_sha256']==value['worker_preflights'][lane] and cfg['physical_gpu']==int(lane),'Fit binding changed: '+candidate)
        value['models'][candidate]=done
    for name,(videos,y) in targets.items():
        value['targets'][name]={'frame_metadata_sha256':p.digest(p.metadata(videos)),
                               'label_sha256':p.sha(config['targets'][name]['labels']),
                               'video_names':[v for v,_ in videos],'video_total_frames':[len(fs) for _,fs in videos]}
    path=p.OUT/'freeze.json'
    if path.exists():support.require(p.read_json(path)==value,'Evaluation freeze changed')
    else:p.write_json(path,value)

def optimizer_regression_gate(p):
    """Atomically commit the disposable three-step CUDA test before any fit."""
    import gpu_regression
    path=p.OUT/'workers'/('gpu'+str(p.LANE))/'optimizer_regression.json'
    pre=p.read_json(p.preflight_path())
    recorded=pre.get('optimizer_regression_sha256')
    if recorded is not None:
        support.require(path.is_file() and p.sha(path)==recorded,'Optimizer gate receipt missing/changed')
    receipt=gpu_regression.run(p)
    if path.exists():support.require(p.read_json(path)==receipt,'Optimizer gate receipt changed')
    else:p.write_json(path,receipt)
    actual=p.sha(path)
    if recorded is None:
        pre['optimizer_regression_sha256']=actual
        p.write_json(p.preflight_path(),pre)
    else:support.require(recorded==actual,'Optimizer gate preflight binding changed')
    print('Three source-only GPU updates: complete G/D/N Adam, N/Arc parameter change and new-updater baseline parity: PASS',flush=True)


def run_phase(phase,lane=None,workers=4,evaluation_batch=16):
    support.require(phase in PHASES and workers>=0 and evaluation_batch==16,'Undeclared runtime options')
    if phase in ('prepare','selection','freeze'):support.require(lane is None,'Coordinator phase cannot have lane')
    else:support.require(lane in QUEUES and os.environ.get('CUDA_VISIBLE_DEVICES')==lane,'Assigned physical GPU visibility required')
    verify_release_files()
    if phase=='prepare':support.prepare_parent()
    support.validate_parent();p=original_pipeline();p.LANE=lane
    if phase=='prepare':
        if not p.prepare():raise SystemExit(20)
        for name in ('source_split.json','source_probe_plan.npy'):
            support.copy_exact(ROOT/'reference'/name,p.OUT/name,support.sha(ROOT/'reference'/name))
        support.validate_prepared();return
    support.validate_prepared();validate_live_data(p)
    if phase in ('selection','freeze'):
        cross_card_receipt(p)
        for card in QUEUES:support.validate_preflight(card)
        if phase=='selection':
            import source_selection
            source_selection.generate_selection()
        else:freeze_evaluation(p,evaluation_batch)
        return
    environment=p.check_runtime()
    if phase=='preflight':
        p.gpu_preflight();optimizer_regression_gate(p);support.validate_preflight(lane);p.status('PREFLIGHT_PASSED');return
    pre=p.read_json(p.preflight_path())
    support.require(pre.get('status')=='PASS' and pre.get('prepared_sha256')==p.sha(p.OUT/'prepared.json') and pre.get('environment')==environment and pre.get('physical_gpu')==int(lane) and pre.get('queue')==QUEUES[lane],'Preflight/runtime changed')
    support.validate_preflight(lane);cross_card_receipt(p)
    if phase=='fit':
        for arm in QUEUES[lane]:
            validate_live_data(p);p.fit(arm,workers)
        p.status('ALL_LANE_FITS_COMPLETED',completed_fits=len(QUEUES[lane]))
    else:
        import probes
        if phase=='probe':
            probes.lane_probe(p,evaluation_batch);p.status('ALL_LANE_PROBES_COMPLETED',completed_probes=len(QUEUES[lane]))
        else:
            import source_selection
            source_selection.validate_source_validation()
            probes.lane_evaluate(p,evaluation_batch);p.status('ALL_LANE_EVALUATIONS_COMPLETED',AUROC_rows=3*len(QUEUES[lane]))
    support.validate_parent();support.validate_prepared();validate_live_data(p)

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase',choices=PHASES,required=True);ap.add_argument('--lane',choices=list(QUEUES))
    ap.add_argument('--workers',type=int,default=4);ap.add_argument('--evaluation-batch',type=int,default=16)
    args=ap.parse_args(argv);run_phase(args.phase,args.lane,args.workers,args.evaluation_batch)

if __name__=='__main__':main()
