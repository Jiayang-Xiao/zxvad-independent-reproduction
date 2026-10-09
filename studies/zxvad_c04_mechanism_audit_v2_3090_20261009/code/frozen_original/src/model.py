"""Single seed17 screening; one intervention family per recipe."""
import math,types
import torch
from torch import nn
from torch.nn import functional as F
import baseline_model as base
from motion_loss import motion_weights
from interventions import coherent_appearance

RECIPES=('B','S25','SR25','C7','M20','MR20','U9','R50','QC03','G25','C3','CM7','M10','F9','R100','QC01','S12','S50','C1','M20W','G00','Tweak')
SEEDS=(17,)
ORDER=[r+'_s17' for r in RECIPES]
DEFAULT={'support':None,'support_shift':False,'skip':None,'motion':None,'blend':None,'real':0.,'compact':0.,'guidance':.5,'appearance':False}
CHANGES={'B':{},'S25':{'support':.25},'SR25':{'support':.25,'support_shift':True},'S12':{'support':.125},'S50':{'support':.5},'C7':{'skip':(7,False)},'C3':{'skip':(3,False)},'CM7':{'skip':(7,True)},'C1':{'skip':(1,False)},'M20':{'motion':(.2,.5,False)},'MR20':{'motion':(.2,.5,True)},'M10':{'motion':(.1,.5,False)},'M20W':{'motion':(.2,.25,False)},'U9':{'blend':'uniform'},'F9':{'blend':'feather'},'R50':{'real':.5},'R100':{'real':1.},'QC03':{'compact':.03},'QC01':{'compact':.01},'G25':{'guidance':.25},'G00':{'guidance':0.},'Tweak':{'appearance':True}}
CONFIG={r:{**DEFAULT,**CHANGES[r]} for r in RECIPES}

def parse(arm):
    recipe,seed=arm.split('_s');seed=int(seed)
    if recipe not in CONFIG or seed!=17:raise ValueError(arm)
    return recipe,seed

@torch.no_grad()
def sparse_support(response,fraction=.25,shift=False):
    if response.ndim!=4 or response.shape[1]!=1 or not torch.isfinite(response).all():raise ValueError('Invalid O response')
    flat=response.flatten(1);k=max(1,math.ceil(fraction*flat.shape[1]));order=flat.argsort(dim=1,descending=True,stable=True)
    mask=torch.zeros_like(flat).scatter_(1,order[:,:k],1).reshape_as(response)
    return torch.roll(mask,(mask.shape[-2]//2,mask.shape[-1]//2),(-2,-1)) if shift else mask

def _sparse_forward(self,x):return sparse_support(base.channel_attention(self.features(x),True),self._fraction,self._shift_support)

class SkipTransform(nn.Module):
    def __init__(self,channels,kernel,masked):
        super().__init__();self.kernel=kernel
        if kernel>1:
            self.depthwise=nn.Conv2d(channels,channels,kernel,padding=kernel//2,groups=channels,bias=False)
            support=torch.ones(1,1,kernel,kernel)
            if masked:support[:,:,kernel//2-1:kernel//2+2,kernel//2-1:kernel//2+2]=0
            self.register_buffer('support',support)
        else:self.depthwise=None
        self.pointwise=nn.Conv2d(channels,channels,1,bias=False)
    def forward(self,x):
        if self.depthwise is not None:x=F.conv2d(x,self.depthwise.weight*self.support,padding=self.kernel//2,groups=x.shape[1])
        return F.relu(self.pointwise(x))

class ConvGenerator(base.Generator):
    def forward(self,x):
        a=self.down1(x);b=self.down2(self.pool(a));c=self.down3(self.pool(b));d,address=self.memory(self.down4(self.pool(c)))
        d=self.dec1(torch.cat((self.skip3(c),self.up1(d)),dim=1));d=self.dec2(torch.cat((self.skip2(b),self.up2(d)),dim=1));d=self.dec3(torch.cat((self.skip1(a),self.up3(d)),dim=1))
        return self.output(d).tanh(),address

def add_skips(g,choice,device):
    g.__class__=ConvGenerator
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator(device='cpu').manual_seed(1017).get_state())
        for i,c in enumerate((64,128,256),1):setattr(g,f'skip{i}',SkipTransform(c,*choice).to(device))
    return g

class MotionLoss(nn.Module):
    def __init__(self,original,choice):super().__init__();self.original=original;self.choice=choice;self.sequence=None
    def forward(self,pred,target,address):
        if self.sequence is None:raise RuntimeError('Missing source sequence')
        _,parts=self.original(pred,target,address);fraction,mass,shift=self.choice
        weights,mask=motion_weights(self.sequence,'shifted' if shift else 'motion',fraction=fraction,local_mass=mass)
        weighted=(weights*(pred-target).square()).mean()
        total=weighted+parts['ssim_loss']+parts['gradient']+.0025*parts['memory_entropy']
        return total,{**parts,'weighted_mse':weighted,'motion_roi_area':mask.mean(),'weight_mean':weights.mean()}

def make_models(arm,device):
    recipe,_=parse(arm);choice=CONFIG[recipe];models,opts,loss,aug=base.make_models('c04',device)
    if choice['support'] is not None:
        o=models['O'];o._fraction=choice['support'];o._shift_support=choice['support_shift'];o.forward=types.MethodType(_sparse_forward,o)
    if choice['skip']:
        add_skips(models['G'],choice['skip'],device);opts['G']=torch.optim.Adam(models['G'].parameters(),lr=.0002,betas=(.5,.999))
    if choice['motion']:loss=MotionLoss(loss,choice['motion'])
    return models,opts,loss,aug

def make_generator(arm,device):
    recipe,_=parse(arm);g=base.Generator('dot').to(device)
    if CONFIG[recipe]['skip']:add_skips(g,CONFIG[recipe]['skip'],device)
    return g

def one_update(clip,donor,arm,step,models,opts,loss,aug,audit=False):
    recipe,_=parse(arm);choice=CONFIG[recipe];handle=None
    if choice['appearance']:clip,donor=coherent_appearance(clip,donor,step)
    if isinstance(loss,MotionLoss):loss.sequence=clip
    if choice['compact']:
        memory=models['G'].memory;memory._screen_query=[]
        def capture(module,inputs):
            if module._screen_query:raise RuntimeError('Unconsumed memory query')
            module._screen_query.append(inputs[0])
        handle=memory.register_forward_pre_hook(capture)
    try:
        if choice['blend'] or choice['real'] or choice['compact'] or choice['guidance']!=.5:
            from variant_update import modified_update
            return modified_update(clip,donor,recipe,step,models,opts,loss,aug,audit=audit)
        return base.one_update(clip,donor,'c04',step,models,opts,loss,aug,audit=audit)
    finally:
        if handle is not None:handle.remove();del models['G'].memory._screen_query
        if isinstance(loss,MotionLoss):loss.sequence=None
