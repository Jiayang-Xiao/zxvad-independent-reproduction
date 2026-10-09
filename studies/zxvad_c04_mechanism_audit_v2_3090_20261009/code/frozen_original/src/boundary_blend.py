"""Single scientific change: opacity placement within the original paste mask."""
import torch
import torch.nn.functional as F

KERNEL_SIZE = 9


@torch.no_grad()
def blend_paste(base, hard, mask, variant):
    if variant == 'baseline':
        return hard, mask
    feather = mask * F.avg_pool2d(mask, KERNEL_SIZE, stride=1,
        padding=KERNEL_SIZE//2, count_include_pad=False)
    if variant == 'feather':
        alpha = feather
    elif variant == 'uniform':
        area = mask.sum(dim=(2,3),keepdim=True)
        strength = feather.sum(dim=(2,3),keepdim=True)/area.clamp_min(1)
        alpha = mask * strength
    else:
        raise ValueError('Unknown blending arm: '+str(variant))
    mixed = base + alpha*(hard-base)
    # Preserve exact pixels at both opacity endpoints, including the hard interior.
    mixed = torch.where(alpha==1,hard,torch.where(alpha==0,base,mixed))
    return mixed, alpha


def check_blending(device):
    """Actual device implementation versus independent neighborhood enumeration."""
    import numpy as np
    masks = np.zeros((6,1,33,35),dtype=np.float32)
    masks[0,0,5:28,6:29] = 1
    masks[1,0,0:18,0:19] = 1
    masks[2,0,16,17] = 1
    masks[3] = 1
    masks[5,0,4:29,5:30] = 1; masks[5,0,14:18,16:19] = 0
    expected = np.zeros_like(masks)
    for b in range(len(masks)):
        for y in range(33):
            for x in range(35):
                expected[b,0,y,x] = masks[b,0,y,x]*masks[b,0,max(0,y-4):y+5,max(0,x-4):x+5].mean()
    mask = torch.from_numpy(masks).to(device)
    base = torch.full((6,3,33,35),-.5,device=device)
    hard = torch.where(mask.bool(),torch.ones_like(base)*.5,base)
    feather, alpha = blend_paste(base,hard,mask,'feather')
    uniform, uniform_alpha = blend_paste(base,hard,mask,'uniform')
    baseline, baseline_alpha = blend_paste(base,hard,mask,'baseline')
    assert baseline is hard and baseline_alpha is mask
    np.testing.assert_allclose(alpha.cpu().numpy(),expected,rtol=1e-6,atol=1e-7)
    torch.testing.assert_close(alpha.sum((2,3)),uniform_alpha.sum((2,3)),rtol=1e-6,atol=1e-4)
    assert torch.equal(alpha[3],mask[3]) and torch.count_nonzero(alpha[4])==0
    assert alpha[0,0,16,17]==1 and 0<alpha[0,0,5,6]<1
    assert torch.equal(feather[0,:,16,17],hard[0,:,16,17])
    assert feather[0,0,5,6]-base[0,0,5,6] < uniform[0,0,5,6]-base[0,0,5,6]
    for actual in (feather,uniform):
        assert torch.isfinite(actual).all()
        assert torch.count_nonzero((actual-base)*(1-mask))==0
        assert torch.all(actual>=torch.minimum(base,hard)) and torch.all(actual<=torch.maximum(base,hard))
    a,_ = blend_paste(base.requires_grad_(),hard.requires_grad_(),mask,'feather')
    assert not a.requires_grad
    print('Blending: neighborhood reference, exact baseline/core/background, mean-opacity control, empty/edge/full masks PASS',flush=True)
