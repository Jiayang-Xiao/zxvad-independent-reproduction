"""Frozen one-seed, two-card matched-control mechanism battery; no torch imports."""
STUDY='ZXVAD_C04_MECHANISM_AUDIT_3090_20261009'
TARGETS={'ped1':7056,'ped2':1962,'avenue':15240}
MODES=('PHOTO','NOISE','SELF','STATIC','MOVE','SHUFFLE')
CONFIG={}
def entry(kind='baseline',nc_source='baseline',guide=.5,cf=None):
    return {'kind':kind,'nc_source':nc_source,'guide':guide,'cf':cf}
for name in ('B0','B1'):CONFIG[name]=entry()
for name,source in [('NC_Y','Y'),('NC_X','X'),('NC_PRED','PRED'),('NC_MIX50','MIX50'),('NC_ORIGIN','ORIGIN')]:CONFIG[name]=entry('nc_origin',source)
CONFIG['NC_OFF']=entry('nc_ablation','baseline',0.)
QUEUES={'0':['B0_s17','NC_Y_s17','NC_X_s17','NC_ORIGIN_s17'],'1':['B1_s17','NC_PRED_s17','NC_MIX50_s17','NC_OFF_s17']}
for lane,area in [('0',.1),('1',.2)]:
    for probability in (.25,.5):
        for mode in MODES:
            recipe=f'CF_{mode}_A{round(area*100)}_P{round(probability*100)}'
            CONFIG[recipe]=entry('counterfactual','baseline',.5,{'mode':mode,'area':area,'p':probability})
            QUEUES[lane].append(recipe+'_s17')
ORDER=[arm for lane in QUEUES for arm in QUEUES[lane]]
BASELINE={'0':'B0_s17','1':'B1_s17'}
SEEDS=(17,)
def parse(arm):
    recipe,seed=arm.rsplit('_s',1);seed=int(seed)
    if recipe=='B':recipe='B0' # disposable original-parity alias, never a planned fit
    if recipe not in CONFIG or seed!=17:raise ValueError('Undeclared arm: '+arm)
    return recipe,seed
def settings(arm):return CONFIG[parse(arm)[0]]
def lane_for(arm):
    for lane,queue in QUEUES.items():
        if arm in queue:return lane
    raise ValueError(arm)
assert len(ORDER)==32 and all(len(q)==16 for q in QUEUES.values()) and len(set(ORDER))==32
READOUTS=('PSNR','NC','NC_GAP','LOCAL','FUSION','PSNR_FROZEN','FUSION_FROZEN')
