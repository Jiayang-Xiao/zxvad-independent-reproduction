"""Exact c04 updater parity on the installed device; disposable source-only check."""
import gc,hashlib
import torch
import baseline_model,model
from common import seed_everything

def state_digest(models,opts):
    h=hashlib.sha256()
    def visit(value):
        if torch.is_tensor(value):
            value=value.detach().cpu().contiguous();h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode());h.update(value.numpy().tobytes())
        elif isinstance(value,dict):
            for k in sorted(value,key=str):h.update(str(k).encode());visit(value[k])
        elif isinstance(value,(list,tuple)):
            for v in value:visit(v)
        else:h.update(repr(value).encode())
    visit({k:m.state_dict() for k,m in models.items()});visit({k:o.state_dict() for k,o in opts.items()})
    visit(torch.get_rng_state());visit(torch.cuda.get_rng_state_all())
    return h.hexdigest()

def check_baseline_parity(clip,donor):
    results=[]
    for wrapped in (False,True):
        seed_everything();m,o,l,a=model.make_models('B',clip.device)
        fn=model.one_update if wrapped else baseline_model.one_update
        metrics=fn(clip,donor,'B' if wrapped else 'c04',0,m,o,l,a,audit=True)
        results.append((metrics,state_digest(m,o)))
        del m,o,l,a;gc.collect()
        if clip.is_cuda:torch.cuda.empty_cache()
    for key,value in results[0][0].items():assert results[1][0][key]==value,'Baseline metric mismatch: '+key
    assert results[0][1]==results[1][1],'Baseline model/Adam/RNG state mismatch'
    return {'status':'PASS','scope':'Bit-identical direct c04 vs B wrapper on same runtime/device/source batch','state_sha256':results[0][1]}
