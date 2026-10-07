"""Real CUDA forward/backward check in the private Blackwell environment."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import check_runtime,write_json,ROOT,seed_everything
import torch
env=check_runtime();seed_everything()
x=torch.randn(2,3,32,32,device='cuda',requires_grad=True)
model=torch.nn.Sequential(torch.nn.Conv2d(3,8,3,padding=1),torch.nn.ReLU(),torch.nn.ConvTranspose2d(8,3,2,2)).cuda()
loss=model(x).square().mean();loss.backward();torch.cuda.synchronize()
assert torch.isfinite(loss) and torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
write_json(ROOT/'outputs'/'gpu_compatibility.json',{'status':'PASS','environment':env,'real_CUDA_forward_backward':True,'synthetic_only_not_a_research_fit':True})
print('Private CUDA12.8 environment and real GPU forward/backward: PASS',flush=True)
