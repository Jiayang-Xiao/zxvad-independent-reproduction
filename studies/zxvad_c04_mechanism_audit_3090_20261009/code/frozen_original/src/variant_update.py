"""Original update with independent blend, positive-input, compactness or guidance switches."""
from baseline_model import *
from model import CONFIG
from interventions import compact_loss
from boundary_blend import blend_paste

def modified_update(clip, donor, recipe, step, models, opts, loss_fn, aug, audit=False):
    G,D,N,O,arc = (models[k] for k in ('G','D','N','O','Arc'))
    choice = CANDIDATES['c04']; settings=CONFIG[recipe]
    x,target = clip[:,:4].flatten(1,2),clip[:,4]
    pred,address = G(x)
    with torch.no_grad():
        omask = O(donor)
        pseudo,mask,boxes = paste(clip[:,0],donor,omask,step)
        hard_support=mask
        if settings['blend']:pseudo,mask=blend_paste(clip[:,0],pseudo,mask,settings['blend'])
    # Freeze parameter gradients for guide networks; BN remains in train mode.
    D.requires_grad_(False); N.requires_grad_(False)
    n_pred,_ = N(pred)
    d_pred,_ = D(pred)
    bb,parts = loss_fn(pred,target,address)
    guidance = .5*F.mse_loss(n_pred,torch.ones_like(n_pred))
    compact=compact_loss(G.memory._screen_query.pop(),G.memory.weight) if settings['compact'] else pred.new_zeros(())
    lg = bb+.05*(.5*F.mse_loss(d_pred,torch.ones_like(d_pred)))+settings['guidance']*guidance+settings['compact']*compact
    if audit:
        grads = torch.autograd.grad(guidance,tuple(G.parameters()),retain_graph=True,allow_unused=True)
        norm = sum(g.abs().sum().item() for g in grads if g is not None)
        assert norm>0 and math.isfinite(norm), 'N-to-G gradient disconnected'
    opts['G'].zero_grad(set_to_none=True); lg.backward(); opts['G'].step()
    D.requires_grad_(True); N.requires_grad_(True)
    real,_ = D(target); fake,_ = D(pred.detach())
    ld = .5*F.mse_loss(real,torch.ones_like(real))+.5*F.mse_loss(fake,torch.zeros_like(fake))
    opts['D'].zero_grad(set_to_none=True); ld.backward(); opts['D'].step()
    normal_input=pred.detach() if not settings['real'] else ((1-settings['real'])*pred.detach()+settings['real']*target.detach())
    normal,normal_feature = N(normal_input)
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
        transformed = aug(((normal_input+1)/2).clamp(0,1)).clamp(0,1)*2-1
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
        assert torch.equal((pseudo-clip[:,0])*(1-hard_support),torch.zeros_like(pseudo))
    opts['N'].zero_grad(set_to_none=True); ln.backward(); opts['N'].step()
    values = {'g':lg,'d':ld,'n':ln,'normalcy':ng,'relative_normalcy':rn,'attention_affirmation':aa,'relative_attention':raa,'guidance_coefficient':pred.new_tensor(settings['guidance']),'query_compact':compact,'real_normal_fraction':pred.new_tensor(settings['real']),**parts}
    result = {key:float(v.detach()) for key,v in values.items()}
    if not all(math.isfinite(v) for v in result.values()): raise RuntimeError('Nonfinite loss')
    result.update(object_mask_area=float(omask.mean()),paste_mask_area=float(mask.mean()),n_attention_mean=float(a.detach().mean()),n_attention_zero_fraction=float((a.detach()==0).float().mean()),n_bn_updates=int(next(m for m in N.modules() if isinstance(m,nn.BatchNorm2d)).num_batches_tracked),boxes=boxes)
    return result
