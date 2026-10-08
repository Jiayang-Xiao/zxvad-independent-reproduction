"""Three fixed training interventions; no target-time modifications."""
import numpy as np
import torch
from torch.nn import functional as F

def tca_parameters(batch,step):
    rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([17,6,int(step)])))
    active=rng.random((batch,1,1,1,1))<.25
    gain=rng.uniform(.9,1.1,(batch,5,3,1,1)).astype(np.float32)[:,:1]
    offset=rng.uniform(-.05,.05,(batch,5,3,1,1)).astype(np.float32)[:,:1]
    return active,gain,offset

def coherent_appearance(clip,donor,step):
    if clip.ndim!=5 or clip.shape[1:3]!=(5,3) or clip.dtype!=torch.float32: raise ValueError('Expected B5CHW float32')
    active,gain,offset=tca_parameters(len(clip),step)
    active=torch.as_tensor(active,device=clip.device)
    gain=torch.as_tensor(gain,device=clip.device); offset=torch.as_tensor(offset,device=clip.device)
    changed=torch.where(active,(clip*gain+2*offset).clamp(-1,1),clip)
    # c04 uses an independent donor; use the RECIPIENT clip affine on that donor.
    changed_donor=torch.where(active[:,0],(donor*gain[:,0]+2*offset[:,0]).clamp(-1,1),donor)
    return changed,changed_donor

def compact_loss(features,weight):
    query=F.normalize(features.permute(0,2,3,1).reshape(-1,features.shape[1]),dim=1,eps=1e-12)
    keys=F.normalize(weight.detach(),dim=1,eps=1e-12)
    with torch.no_grad():
        # argmax returns the first index on ties; no topk tie ambiguity.
        first=F.linear(query.detach(),keys).argmax(dim=1)
    return F.mse_loss(query,keys[first])
