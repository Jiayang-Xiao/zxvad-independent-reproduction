"""Frozen c04 models, with explicit T/N/Q flags and a narrow updater diff."""
from baseline_model import *
import baseline_model as baseline
from interventions import coherent_appearance,compact_loss

CANDIDATES={name:{'attention':'hidden_normalized','memory':'dot','T':'T' in name,'N':'N' in name,'Q':'Q' in name} for name in ('B','T','N','Q','TN','TQ','NQ','TNQ')}

def make_models(candidate,device):
    if candidate not in CANDIDATES: raise ValueError(candidate)
    return baseline.make_models('c04',device)

def one_update(clip,donor,candidate,step,models,opts,loss_fn,aug,audit=False):
    choice=CANDIDATES[candidate]
    if choice['T']: clip,donor=coherent_appearance(clip,donor,step)
    if not (choice['N'] or choice['Q']):
        result=baseline.one_update(clip,donor,'c04',step,models,opts,loss_fn,aug,audit=audit)
        result.update(guidance_coefficient=.5,query_compact=0.)
        return result
    handle=None
    if choice['Q']:
        mem=models['G'].memory; mem._combo_query=[]
        def capture(module,inputs):
            if module._combo_query: raise RuntimeError('Unconsumed query')
            module._combo_query.append(inputs[0])
        handle=mem.register_forward_pre_hook(capture)
    try:
        return _modified_update(clip,donor,candidate,step,models,opts,loss_fn,aug,audit=audit)
    finally:
        if handle is not None:
            handle.remove(); del models['G'].memory._combo_query

def _modified_update(clip, donor, candidate, step, models, opts, loss_fn, aug, audit=False):
    G,D,N,O,arc = (models[k] for k in ('G','D','N','O','Arc'))
    choice = CANDIDATES[candidate]
    x,target = clip[:,:4].flatten(1,2),clip[:,4]
    pred,address = G(x)
    with torch.no_grad():
        omask = O(donor)
        pseudo,mask,boxes = paste(clip[:,0],donor,omask,step)
    # Freeze parameter gradients for guide networks; BN remains in train mode.
    D.requires_grad_(False); N.requires_grad_(False)
    n_pred,_ = N(pred)
    d_pred,_ = D(pred)
    bb,parts = loss_fn(pred,target,address)
    guidance = .5*F.mse_loss(n_pred,torch.ones_like(n_pred))
    coeff=.45005 if choice['N'] else .5
    compact=compact_loss(models['G'].memory._combo_query.pop(),models['G'].memory.weight) if choice['Q'] else pred.new_zeros(())
    lg = bb+.05*(.5*F.mse_loss(d_pred,torch.ones_like(d_pred)))+coeff*guidance+.1*compact
    if audit:
        grads = torch.autograd.grad(guidance,tuple(G.parameters()),retain_graph=True,allow_unused=True)
        norm = sum(g.abs().sum().item() for g in grads if g is not None)
        assert norm>0 and math.isfinite(norm), 'N-to-G gradient disconnected'
    opts['G'].zero_grad(set_to_none=True); lg.backward(); opts['G'].step()
    D.requires_grad_(True); N.requires_grad_(True)
    real,_ = D(target); fake,_ = D(pred.detach())
    ld = .5*F.mse_loss(real,torch.ones_like(real))+.5*F.mse_loss(fake,torch.zeros_like(fake))
    opts['D'].zero_grad(set_to_none=True); ld.backward(); opts['D'].step()
    normal,normal_feature = N(pred.detach())
    abnormal,abnormal_feature = N(pseudo)
    a = classifier_attention(normal,normal_feature,choice['attention'])
    ap = classifier_attention(abnormal,abnormal_feature,choice['attention'])
    ng = .5*F.mse_loss(normal,torch.ones_like(normal))+.5*F.mse_loss(abnormal,torch.zeros_like(abnormal))
    rn = .5*F.mse_loss(normal-abnormal.mean(),torch.ones_like(normal))+.5*F.mse_loss(abnormal-normal.mean(),-torch.ones_like(abnormal))
    resized_mask = F.interpolate(mask,a.shape[-2:],mode='nearest')
    aa = .5*F.mse_loss(a,torch.ones_like(a))+.5*F.mse_loss(ap,resized_mask)
    devices = [clip.device.index] if clip.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(17+step+700001)
        if clip.is_cuda: torch.cuda.manual_seed_all(17+step+700001)
        transformed = aug(((pred.detach()+1)/2).clamp(0,1)).clamp(0,1)*2-1
    aug_logits,aug_feature = N(transformed)
    av = classifier_attention(aug_logits,aug_feature,choice['attention'])
    labels = torch.cat((torch.ones(2*clip.shape[0],device=clip.device,dtype=torch.long),torch.zeros(clip.shape[0],device=clip.device,dtype=torch.long)))
    raa = arc(torch.cat((a.flatten(1),av.flatten(1),ap.flatten(1))),labels)
    ln = ng+.01*rn+aa+raa
    if audit:
        grads = torch.autograd.grad(ln,tuple(G.parameters()),retain_graph=True,allow_unused=True)
        assert all(g is None for g in grads), 'N update connected to G'
        assert not pseudo.requires_grad and not mask.requires_grad
        assert all(p.grad is None for p in O.parameters())
        assert torch.equal((pseudo-clip[:,0])*(1-mask),torch.zeros_like(pseudo))
    opts['N'].zero_grad(set_to_none=True); ln.backward(); opts['N'].step()
    values = {'g':lg,'d':ld,'n':ln,'normalcy':ng,'relative_normalcy':rn,'attention_affirmation':aa,'relative_attention':raa,'guidance_raw':guidance,'guidance_coefficient':pred.new_tensor(coeff),'query_compact':compact,**parts}
    result = {key:float(v.detach()) for key,v in values.items()}
    if not all(math.isfinite(v) for v in result.values()): raise RuntimeError('Nonfinite loss')
    result.update(object_mask_area=float(omask.mean()),paste_mask_area=float(mask.mean()),n_attention_mean=float(a.detach().mean()),n_attention_zero_fraction=float((a.detach()==0).float().mean()),n_bn_updates=int(next(m for m in N.modules() if isinstance(m,nn.BatchNorm2d)).num_batches_tracked),boxes=boxes)
    return result
