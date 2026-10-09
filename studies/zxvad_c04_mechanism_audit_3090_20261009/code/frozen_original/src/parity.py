"""Original c04, direct B wrapper and no-op variant updater equality."""
import gc,hashlib
import torch
import baseline_model,model
from common import seed_everything

def state_digest(models,opts):
    h=hashlib.sha256()
    def visit(v):
        if torch.is_tensor(v):
            v=v.detach().cpu().contiguous();h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode());h.update(v.numpy().tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=str):h.update(str(k).encode());visit(v[k])
        elif isinstance(v,(list,tuple)):
            for x in v:visit(x)
        else:h.update(repr(v).encode())
    visit({k:m.state_dict() for k,m in models.items()});visit({k:o.state_dict() for k,o in opts.items()});visit(torch.get_rng_state())
    if torch.cuda.is_available():visit(torch.cuda.get_rng_state_all())
    return h.hexdigest()

def check_parity(clip,donor):
    from variant_update import modified_update
    results=[]
    for mode in ('original','wrapped','variant_noop'):
        seed_everything();m,o,l,a=model.make_models('B_s17',clip.device)
        if mode=='original':r=baseline_model.one_update(clip,donor,'c04',0,m,o,l,a,audit=True)
        elif mode=='wrapped':r=model.one_update(clip,donor,'B_s17',0,m,o,l,a,audit=True)
        else:r=modified_update(clip,donor,'B',0,m,o,l,a,audit=True)
        results.append((r,state_digest(m,o)));del m,o,l,a;gc.collect()
        if clip.is_cuda:torch.cuda.empty_cache()
    for r,h in results[1:]:
        assert h==results[0][1],'Original model/Adam/RNG mismatch'
        assert all(r[k]==v for k,v in results[0][0].items()),'Original metrics mismatch'
    return {'status':'PASS','state_sha256':results[0][1],'scope':'Full update exact original/B/no-op-variant model,Adam,metrics,RNG'}
