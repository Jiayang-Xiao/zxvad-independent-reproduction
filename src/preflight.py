"""Source-only baseline CUDA update checks and fixed data-layout validation."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from baseline_zxvad import (ArcFaceLoss, FutureFrameGenerator, NormalcyAugmentation,
    NormalcyClassifier, PatchDiscriminator, UntrainedAnomalySynthesizer)
from common import TARGETS, EXPECTED, code_identity, digest, frames, identity, sample_batch, save_json, sha
from train import one_update


def checks(device, videos):
    random.seed(17); np.random.seed(17); torch.manual_seed(17); torch.cuda.manual_seed_all(17)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    models = {'G': FutureFrameGenerator(4, 2000).to(device), 'D': PatchDiscriminator().to(device),
        'N': NormalcyClassifier().to(device), 'O': UntrainedAnomalySynthesizer('resnet50').to(device),
        'arcface': ArcFaceLoss(64).to(device)}
    for key, model in models.items(): model.eval() if key == 'O' else model.train()
    opts = {'G': torch.optim.Adam(models['G'].parameters(), lr=2e-4, betas=(.5,.999)),
        'D': torch.optim.Adam(models['D'].parameters(), lr=2e-5, betas=(.5,.999)),
        'N': torch.optim.Adam(list(models['N'].parameters())+list(models['arcface'].parameters()), lr=2e-5, betas=(.5,.999))}
    before = {key: next(model.parameters()).detach().clone() for key, model in models.items()}
    batch = torch.from_numpy(sample_batch(videos, 0)).to(device)
    losses = one_update(batch, models, opts, NormalcyAugmentation(), 'baseline', audit=True)
    assert all(torch.isfinite(x) for x in losses)
    for key in ('G','D','N','arcface'):
        assert not torch.equal(before[key], next(models[key].parameters()).detach()), key+' did not update'
    assert torch.equal(before['O'], next(models['O'].parameters()).detach())
    assert all(not p.requires_grad for p in models['O'].parameters())
    n_bn = [int(x.num_batches_tracked.item()) for x in models['N'].modules() if isinstance(x,torch.nn.BatchNorm2d)]
    assert n_bn and set(n_bn)=={5}, 'Legacy N must preserve five train-mode forwards'
    return {'source_sample_step':0, 'first_step_losses_g_d_n_mse':[float(x.item()) for x in losses],
        'N_BN_updates_per_step':5, 'checks':['finite full source update', 'N-to-G gradient connected',
        'N update detached from G', 'G/D/N/ArcFace parameters update', 'O frozen', 'legacy N BN five forwards']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source',required=True); ap.add_argument('--target-root',required=True)
    ap.add_argument('--output',required=True); ap.add_argument('--device',default='cuda:0')
    args=ap.parse_args()
    source,root=Path(args.source).resolve(),Path(args.target_root).resolve()
    if not source.is_dir(): raise FileNotFoundError('Source frames missing: '+str(source))
    videos=frames(source)
    if len(videos)!=330 or any(len(fs)<5 for _,fs in videos): raise RuntimeError('Expected 330 usable ShanghaiTech normal videos')
    for target in TARGETS:
        path=(root/target/'testing'/'frames').resolve()
        if source==path or source in path.parents or path in source.parents: raise RuntimeError('Source/target overlap')
        catalog=frames(path); y=np.load(root/f'frame_labels_{target}.npy',allow_pickle=False).reshape(-1)
        if (len(catalog),sum(len(fs)-4 for _,fs in catalog))!=EXPECTED[target]: raise RuntimeError('Canonical frame mismatch: '+target)
        if sum(len(fs) for _,fs in catalog)!=len(y) or not np.isin(y,[0,1]).all(): raise RuntimeError('Label mismatch: '+target)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    provenance={'study':'ZXVAD_INDEPENDENT_LEGACY_V1','code_sha256':code_identity(),
        'recipe_sha256':sha(Path(__file__).parent/'protocol.json'), 'source_metadata_sha256':digest(identity(videos)),
        'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(args.device)}
    record=Path(args.output)/'preflight.json'
    if record.exists():
        old=json.loads(record.read_text())
        if old.get('status')!='PASS' or any(old.get(k)!=v for k,v in provenance.items()): raise RuntimeError('Preflight identity changed')
        print('Existing matching preflight PASS reused.',flush=True); return
    checked=checks(args.device,videos)
    save_json(record,{'status':'PASS',**provenance,**checked,'research_training_executed':False})
    print('Source update, gradients, legacy BN5 and data layout: PASS',flush=True)

if __name__=='__main__': main()
