"""Freeze complete heldout-source evidence and deterministic source-only choices."""
from pathlib import Path
import spec
from study_support import read,write,sha,require,digest
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'
ARTIFACTS=('source_probe.json','source_probe.npz','calibration.json','completed.json')
RULES={'clean_mse_max_ratio':1.10,'time_sensitivity_min_ratio':.80,
       'source_move_AUROC_min_delta':.005,'nc_origin_bias_min_reduction':.02,
       'matched_CF_controls':['PHOTO','NOISE','SELF','STATIC','SHUFFLE'],
       'max_choices_per_family':1,'tie_break':'source_MOVE_AUROC descending, then arm lexical',
       'scope':'Source synthetic mechanism preference only; does not imply real-target superiority'}

def validation_value():
    value={'study':spec.STUDY,'prepared_sha256':sha(OUT/'prepared.json'),
           'source_split_sha256':sha(OUT/'source_split.json'),
           'source_probe_plan_sha256':sha(OUT/'source_probe_plan.npy'),'artifacts':{}}
    for arm in spec.ORDER:
        dst=OUT/arm;done=read(dst/'completed.json');cfg=read(dst/'config.json');probe=read(dst/'source_probe.json');cal=read(dst/'calibration.json')
        require(done['step']==5000 and done['config_hash']==digest(cfg),'Final fit required: '+arm)
        b=probe['binding']
        require(b['study']==spec.STUDY and b['candidate']==arm and b['checkpoint_sha256']==done['checkpoint_sha256']
                and b['prepared_sha256']==value['prepared_sha256'] and b['source_split_sha256']==value['source_split_sha256']
                and b['source_probe_plan_sha256']==value['source_probe_plan_sha256'] and b['source_only'] is True and b['target_used'] is False,
                'Source probe binding changed: '+arm)
        require(cal['binding']==b and probe['scores_sha256']==sha(dst/'source_probe.npz') and probe['calibration_sha256']==sha(dst/'calibration.json'),
                'Source artifact binding changed: '+arm)
        value['artifacts'][arm]={name:sha(dst/name) for name in ARTIFACTS}
    return value

def validate_source_validation():
    path=OUT/'source_validation.json';value=read(path)
    require(value==validation_value(),'Heldout source validation changed')
    selection=OUT/'source_selection.json';freeze=OUT/'freeze.json'
    if selection.exists():require(read(selection)['source_validation_sha256']==sha(path),'Source selection evidence changed')
    if freeze.exists():
        f=read(freeze)
        require(f['source_validation_sha256']==sha(path) and f['source_selection_sha256']==sha(selection),'Frozen source evidence/choice changed')
    return value

def selection_value():
    records={arm:read(OUT/arm/'source_probe.json') for arm in spec.ORDER};candidates={};eligible=[]
    for arm in spec.ORDER:
        settings=spec.settings(arm);baseline=spec.BASELINE[spec.lane_for(arm)]
        cur,b=records[arm],records[baseline]
        ratio=cur['clean_mse']/max(b['clean_mse'],1e-12)
        time_ratio=cur['time_sensitivity']/max(b['time_sensitivity'],1e-12)
        score=cur['source_MOVE_AUROC'];reasons=[];controls={}
        functional=(settings['kind']=='nc_origin' and settings['nc_source']!='ORIGIN') or (settings['kind']=='counterfactual' and settings['cf']['mode']=='MOVE')
        if not functional:reasons.append('Baseline, diagnostic or matched control; not a functional candidate')
        if ratio>RULES['clean_mse_max_ratio']:reasons.append('Clean prediction guard failed')
        if time_ratio<RULES['time_sensitivity_min_ratio']:reasons.append('Temporal dependency guard failed')
        if score<b['source_MOVE_AUROC']+RULES['source_move_AUROC_min_delta']:reasons.append('Insufficient source MOVE AUROC gain over same-card baseline')
        origin=abs(cur['origin']['no_event_real0_pred1_AUROC']-.5)
        base_origin=abs(b['origin']['no_event_real0_pred1_AUROC']-.5)
        if functional and settings['kind']=='nc_origin' and origin>base_origin-RULES['nc_origin_bias_min_reduction']:
            reasons.append('Origin-bias reduction guard failed')
        if settings['kind']=='counterfactual' and settings['cf']['mode']=='MOVE':
            for mode in RULES['matched_CF_controls']:
                other=arm.replace('CF_MOVE_','CF_'+mode+'_')
                controls[other]=score-records[other]['source_MOVE_AUROC']
                if controls[other]<RULES['source_move_AUROC_min_delta']:reasons.append('Does not exceed matched '+mode+' repair control on source MOVE AUROC')
        ok=not reasons
        if ok:eligible.append(arm)
        candidates[arm]={'eligible':ok,'reasons':reasons,'same_card_baseline':baseline,
                         'clean_mse_ratio':ratio,'time_sensitivity_ratio':time_ratio,
                         'source_MOVE_AUROC':score,'origin_bias':origin,'baseline_origin_bias':base_origin,
                         'matched_source_AUROC_deltas':controls}
    chosen=[]
    for kind in ('nc_origin','counterfactual'):
        subset=[arm for arm in eligible if spec.settings(arm)['kind']==kind]
        if subset:chosen.append(sorted(subset,key=lambda arm:(-candidates[arm]['source_MOVE_AUROC'],arm))[0])
    return {'study':spec.STUDY,'metric':'AUROC only','source_validation_sha256':sha(OUT/'source_validation.json'),
            'target_used':False,'rules':RULES,'chosen_arms':chosen,'eligible_arms':eligible,'candidates':candidates}

def generate_selection():
    path=OUT/'source_validation.json';chosen=OUT/'source_selection.json'
    targets_exist=any((OUT/arm/(target+'.npz')).exists() for arm in spec.ORDER for target in spec.TARGETS)
    require(not targets_exist or (path.exists() and chosen.exists()),'Cannot create source choice after target scores exist')
    value=validation_value()
    if path.exists():require(read(path)==value,'Source validation identity changed')
    else:write(path,value)
    selection=selection_value()
    if chosen.exists():require(read(chosen)==selection,'Source choice cannot be overwritten')
    else:write(chosen,selection)
    validate_source_validation()
    print('Source-only choice frozen:',selection['chosen_arms'],'(empty is allowed)',flush=True)
    return selection

if __name__=='__main__':generate_selection()
