"""Paper-guided independent zxVAD candidates. Not the authors' unavailable code."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

CANDIDATES = {
    'c01': {'attention': 'final_raw', 'memory': 'cosine'},
    'c02': {'attention': 'final_normalized', 'memory': 'cosine'},
    'c03': {'attention': 'hidden_normalized', 'memory': 'cosine'},
    'c04': {'attention': 'hidden_normalized', 'memory': 'dot'},
}

class DoubleConv(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(nn.Conv2d(cin,cout,3,padding=1), nn.ReLU(), nn.Conv2d(cout,cout,3,padding=1), nn.ReLU())

class Memory(nn.Module):
    def __init__(self, mode, slots=2000, channels=512):
        super().__init__()
        self.mode = mode
        self.weight = nn.Parameter(torch.empty(slots,channels))
        nn.init.uniform_(self.weight, -1/math.sqrt(channels), 1/math.sqrt(channels))

    def forward(self, x):
        b,c,h,w = x.shape
        query = x.permute(0,2,3,1).reshape(-1,c)
        if self.mode == 'cosine':
            logits = F.linear(F.normalize(query,dim=1), F.normalize(self.weight,dim=1))
        else:
            logits = F.linear(query,self.weight)
        address = logits.softmax(dim=1)
        delta = address - .0005
        address = F.relu(delta)*address/(delta.abs()+1e-12)
        address = F.normalize(address,p=1,dim=1)
        read = F.linear(address,self.weight.t())
        return read.reshape(b,h,w,c).permute(0,3,1,2).contiguous(), address

class Generator(nn.Module):
    def __init__(self, memory='cosine'):
        super().__init__()
        self.down1, self.down2 = DoubleConv(12,64), DoubleConv(64,128)
        self.down3, self.down4 = DoubleConv(128,256), DoubleConv(256,512)
        self.pool = nn.MaxPool2d(2)
        self.memory = Memory(memory)
        self.up1, self.up2, self.up3 = nn.ConvTranspose2d(512,256,2,2), nn.ConvTranspose2d(256,128,2,2), nn.ConvTranspose2d(128,64,2,2)
        self.dec1, self.dec2, self.dec3 = DoubleConv(512,256), DoubleConv(256,128), DoubleConv(128,64)
        self.output = nn.Conv2d(64,3,3,padding=1)

    def forward(self, x):
        a = self.down1(x)
        b = self.down2(self.pool(a))
        c = self.down3(self.pool(b))
        d, address = self.memory(self.down4(self.pool(c)))
        d = self.dec1(torch.cat((c,self.up1(d)),dim=1))
        d = self.dec2(torch.cat((b,self.up2(d)),dim=1))
        d = self.dec3(torch.cat((a,self.up3(d)),dim=1))
        return self.output(d).tanh(), address

class PatchGAN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Conv2d(3,64,4,2,1),nn.LeakyReLU(.2)]
        for cin,cout,stride in [(64,128,2),(128,256,2),(256,512,1)]:
            layers += [nn.Conv2d(cin,cout,4,stride,1,bias=False),nn.BatchNorm2d(cout),nn.LeakyReLU(.2)]
        self.hidden = nn.Sequential(*layers)
        self.score = nn.Conv2d(512,1,4,1,1)

    def forward(self, x):
        hidden = self.hidden(x)
        return self.score(hidden), hidden

def channel_attention(feature, normalized):
    attention = feature.sum(dim=1,keepdim=True)
    if normalized:
        attention = attention.relu()
        attention = attention/attention.amax(dim=(2,3),keepdim=True).clamp_min(1e-8)
    return attention

def classifier_attention(logits, hidden, mode):
    return channel_attention(hidden if mode=='hidden_normalized' else logits, mode!='final_raw')

class ObjectExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import resnet50
        self.features = nn.Sequential(*list(resnet50(weights=None).children())[:-2])
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        return (channel_attention(self.features(x),True)>.1).float()

@torch.no_grad()
def paste(base, donor, small_mask, step, seed=17):
    """Clipped CutMix corners; same paste draws across candidates and resume."""
    b,_,h,w = base.shape
    masks = F.interpolate(small_mask,(h,w),mode='nearest')
    rng = np.random.RandomState((seed+1000003*(step+1))%(2**32))
    frames, labels, boxes = [], [], []
    for i in range(b):
        beta,cx,cy = rng.uniform(),rng.uniform(0,w),rng.uniform(0,h)
        bw,bh = w*math.sqrt(1-beta),h*math.sqrt(1-beta)
        l,r = int(max(0,cx-bw/2)),int(min(w,cx+bw/2))
        t,d = int(max(0,cy-bh/2)),int(min(h,cy+bh/2))
        r,d = max(l+1,r),max(t+1,d)
        obj = F.interpolate((donor[i:i+1]*masks[i:i+1]),(d-t,r-l),mode='bilinear',align_corners=False)
        mask = F.interpolate(masks[i:i+1],(d-t,r-l),mode='nearest')
        result = base[i:i+1].clone()
        result[:,:,t:d,l:r] = torch.where(mask.bool(),obj,result[:,:,t:d,l:r])
        label = torch.zeros_like(masks[i:i+1])
        label[:,:,t:d,l:r] = mask
        frames.append(result); labels.append(label); boxes.append([l,t,r,d])
    return torch.cat(frames),torch.cat(labels),boxes

class BackboneLoss(nn.Module):
    def __init__(self):
        super().__init__()
        coords = torch.arange(11,dtype=torch.float32)-5
        g = torch.exp(-coords.square()/(2*1.5**2)); g = g/g.sum()
        self.register_buffer('window',(g[:,None]*g[None,:])[None,None].expand(3,1,11,11).contiguous())

    def forward(self, pred, target, address):
        x,y = (pred+1)/2,(target+1)/2
        mx,my = F.conv2d(x,self.window,groups=3),F.conv2d(y,self.window,groups=3)
        vx = F.conv2d(x*x,self.window,groups=3)-mx*mx
        vy = F.conv2d(y*y,self.window,groups=3)-my*my
        xy = F.conv2d(x*y,self.window,groups=3)-mx*my
        sim = ((2*mx*my+.01**2)*(2*xy+.03**2)/((mx*mx+my*my+.01**2)*(vx+vy+.03**2))).mean()
        def gradients(z):
            dx = F.pad(z,(0,1,0,0))[:,:,:,1:]-z
            dy = z-F.pad(z,(0,0,0,1))[:,:,1:,:]
            return dx.abs(),dy.abs()
        px,py = gradients(pred); tx,ty = gradients(target)
        gd = ((px-tx).abs()+(py-ty).abs()).mean()
        mse = F.mse_loss(pred,target)
        entropy = -(address*(address+1e-12).log()).sum(1).mean()
        return mse+1-sim+gd+.0025*entropy, {'mse':mse,'ssim_loss':1-sim,'gradient':gd,'memory_entropy':entropy}

def make_models(candidate, device):
    from pytorch_metric_learning.losses import ArcFaceLoss
    from kornia.augmentation import AugmentationSequential,ColorJitter,RandomAffine,RandomPerspective
    choice = CANDIDATES[candidate]
    # Construct G/D/N/O before the size-dependent ArcFace matrix.
    models = {'G':Generator(choice['memory']),'D':PatchGAN(),'N':PatchGAN(),'O':ObjectExtractor()}
    models['Arc'] = ArcFaceLoss(num_classes=2,embedding_size=961 if choice['attention']=='hidden_normalized' else 900,margin=28.6,scale=64)
    models = {name:model.to(device) for name,model in models.items()}
    models['O'].eval()
    loss = BackboneLoss().to(device)
    aug = AugmentationSequential(ColorJitter(.1,.1,.1,.1,p=1),RandomAffine(degrees=360,p=1),RandomPerspective(distortion_scale=.2,p=1),data_keys=['input']).to(device)
    opts = {'G':torch.optim.Adam(models['G'].parameters(),lr=.0002,betas=(.5,.999)),
            'D':torch.optim.Adam(models['D'].parameters(),lr=.00002,betas=(.5,.999)),
            'N':torch.optim.Adam(list(models['N'].parameters())+list(models['Arc'].parameters()),lr=.00002,betas=(.5,.999))}
    return models,opts,loss,aug

def one_update(clip, donor, candidate, step, models, opts, loss_fn, aug, audit=False):
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
    lg = bb+.05*(.5*F.mse_loss(d_pred,torch.ones_like(d_pred)))+.5*guidance
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
    values = {'g':lg,'d':ld,'n':ln,'normalcy':ng,'relative_normalcy':rn,'attention_affirmation':aa,'relative_attention':raa,**parts}
    result = {key:float(v.detach()) for key,v in values.items()}
    if not all(math.isfinite(v) for v in result.values()): raise RuntimeError('Nonfinite loss')
    result.update(object_mask_area=float(omask.mean()),paste_mask_area=float(mask.mean()),n_attention_mean=float(a.detach().mean()),n_attention_zero_fraction=float((a.detach()==0).float().mean()),n_bn_updates=int(next(m for m in N.modules() if isinstance(m,nn.BatchNorm2d)).num_batches_tracked),boxes=boxes)
    return result
