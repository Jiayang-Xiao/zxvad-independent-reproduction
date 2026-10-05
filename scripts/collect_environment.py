"""Record the current host environment separately from historical-run evidence."""
import argparse
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',required=True); args=ap.parse_args()
    result={'scope':'publication_host_current_environment_not_historical_run',
        'python':platform.python_version(),'system':platform.system(),'machine':platform.machine(),'packages':{}}
    for name in ('torch','torchvision','numpy','Pillow','kornia'):
        try: result['packages'][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: result['packages'][name]=None
    try:
        import torch
        result['torch_cuda_build']=torch.version.cuda
        result['cuda_available']=torch.cuda.is_available()
        result['devices']=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception as exc: result['torch_import_error']=type(exc).__name__+': '+str(exc)
    try:
        got=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],
            capture_output=True,text=True,timeout=15,check=True)
        result['nvidia_smi']=got.stdout.strip().splitlines()
    except (OSError,subprocess.SubprocessError): result['nvidia_smi']=None
    path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('Current host environment recorded: '+str(path))

if __name__=='__main__': main()
